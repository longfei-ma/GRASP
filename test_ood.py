import argparse
import sys
import os
import time
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, coalesce

from logger import Logger
from dataset import load_dataset
from baselines import *
from grasp import GRASP
#from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import rand_splits, eval_acc, eval_rocauc, to_sparse_tensor, evaluate_ood
from parse import parse_method, parser_add_main_args
from hyparams import hparams
import faulthandler; faulthandler.enable()


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
# args_dict = vars(args)
if args.ood_type=='label' and args.dataset in hparams:
    for hname, v in hparams[args.dataset][args.method].items():
        setattr(args, hname, v)
print(args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args.dataset, args.ood_type)
edge_index = dataset_ind.edge_index
num_nodes = dataset_ind.num_nodes
ood_idx = dataset_ood_te.node_idx
c = dataset_ind.y.max().item() + 1
d = dataset_ind.num_node_features
model = parse_method(args, dataset_ind, num_nodes, c, d)

if args.dataset not in ['arxiv-year', 'snap-patents']:
    edge_index = to_undirected(edge_index)

if args.ood_type == 'label':
    ood_type = 'semantic'
else:
    ood_type = 'covariate'
if args.rand_split or args.dataset in ['ogbn-proteins']:
    #generate split to dump to local file
    split_idx_lst = [rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
    
    torch.save(split_idx_lst, f'./data/splits/{ood_type}/{args.dataset}-splits.pt')
else:
    split_idx_lst = torch.load(f'./data/splits/{ood_type}/{args.dataset}-splits.pt')

print(f"num nodes {num_nodes} | num classes {c} | num node feats {d}")
print('MODEL:', model)

# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

def estimate_id_to_ood_measure(edge_index, train_idx, num_nodes, test_id, test_ood, logit, device, args):
    if args.tau2 == 100:
        return train_idx.tolist()
    K = int(args.tau2/100 * len(train_idx))
    if args.st == 'random':
        return np.random.choice(train_idx, K, replace=False).tolist()
    
    test_nodes = torch.concat([test_id, test_ood])
    probs = torch.softmax(logit, dim=-1)
    test_probs = probs[test_nodes]

    p = args.tau1
    values, indices = test_probs.max(dim=1)
    if args.st == 'test':
        K = int(args.tau2/100 * len(test_nodes))
        return test_nodes[np.argpartition(values, kth=-K)[-K:]].tolist()

    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    test_ood_pred = test_nodes[mask]
    thresholds2 = np.percentile(values, 100-p)
    mask = values > thresholds2
    test_id_pred = test_nodes[mask]

    #calculate metric to select G
    row, col = edge_index
    N = num_nodes
    value = torch.ones_like(row)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[test_id_pred] = 1
    ood_count[test_ood_pred] = 1
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(device)
    id_count = matmul(adj, id_count)
    ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[train_idx]/(ood_count[train_idx]+1)
    
    #select the top big K
    if args.st == 'top':
        return train_idx[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    #select the top small K
    elif args.st == 'low':
        return train_idx[np.argpartition(metrics, kth=K)[: K]].tolist()
    

logger = Logger(args.runs, args)
model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
model_dir = f'checkpoints/{ood_type}/{model_path}/{args.method}'
print(model_dir)
save_dir = f'metrics/{args.dataset}'

ood = eval(args.ood)(args)
### Testing ###
durations = []
for run in range(args.runs):
    t = time.time()
    print(f'----start time: {t}')
    split_idx = split_idx_lst[run]

    if glob.glob(f'{model_dir}/logit{run}*.pt'):
        print(f'logit ckpt{run} exists, load it ...')
        ckpt = glob.glob(f'{model_dir}/logit{run}*.pt')[0]
        logit = torch.load(ckpt, map_location='cpu')
    else:
        checkpoint = glob.glob(f'{model_dir}/model{run}*.pt')[0]
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        model.eval()

        with torch.no_grad():
            logit = model(dataset_ind)
    
    if args.ood in ['MSP', 'Energy', 'ODIN']:
        scores = ood.detect(logit)
    elif args.ood == 'GNNSafe':
        scores = ood.detect(logit, dataset_ind.edge_index, args)
    elif args.ood == 'Mahalanobis':
        scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]), torch.concat([split_idx['test'], dataset_ood_te.node_idx]), dataset_ind.y)
    elif args.ood == 'KNN':
        score_ckpt = f'{model_dir}/score{run}.pt'
        if os.path.exists(score_ckpt):
            scores = torch.load(score_ckpt, map_location='cpu')
        else:
            scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]))
            torch.save(scores, score_ckpt)
    elif args.ood == 'GRASP':
        selected_nodes = estimate_id_to_ood_measure(edge_index, torch.concat([split_idx['train'], split_idx['valid']]), num_nodes, split_idx['test'], ood_idx, logit.cpu(), device, args)
        scores = ood.detect(logit, dataset_ind.edge_index, selected_nodes, args)

    scores = scores.to(device)
    iid_score = scores[split_idx['test']]
    ood_score = scores[ood_idx]
    result = evaluate_ood(iid_score, ood_score)[:-1]
    logger.add_result(run, result)
    durations.append(time.time()-t)

ood_name = args.ood
print(f'======={ood_name}, time = {np.array(durations).mean():.5f}==========')

### Save results ###
result = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{sub_dataset}" + f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}\n")

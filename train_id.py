import argparse
import sys
import os
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_dataset
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import rand_splits, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor, load_fixed_splits
from parse import parse_method, parser_add_main_args
from hyparams import hparams
import faulthandler; faulthandler.enable()


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)
args_dict = vars(args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args.dataset, args.ood_type)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_te.y.shape) == 1:
    dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

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

if args.dataset == 'ogbn-proteins':
    if args.method == 'mlp' or args.method == 'cs':
        dataset_ind.graph['node_feat'] = scatter(dataset_ind.graph['edge_feat'], dataset_ind.edge_index[0],
            dim=0, dim_size=dataset_ind.graph['num_nodes'], reduce='mean')
    else:
        dataset_ind.graph['edge_index'] = to_sparse_tensor(dataset_ind.graph['edge_index'],
            dataset_ind.graph['edge_feat'], dataset_ind.graph['num_nodes'])
        dataset_ind.graph['node_feat'] = dataset_ind.graph['edge_index'].mean(dim=1)
        dataset_ind.graph['edge_index'].set_value_(None)
    dataset_ind.graph['edge_feat'] = None

n = dataset_ind.num_nodes
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.num_node_features

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset_ind.edge_index = to_undirected(dataset_ind.edge_index)

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###

dataset_ind.x, dataset_ind.y = dataset_ind.x.to(device), dataset_ind.y.to(device)
dataset_ind.edge_index = dataset_ind.edge_index.to(device)
model = parse_method(args, dataset_ind, n, c, d, device)
model = model.to(device)


# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)
model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
model_dir = f'checkpoints/{ood_type}/{model_path}/{args.method}'
print(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.train()
print('MODEL:', model)

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('-inf')
    for epoch in range(args.epochs):
        model.train()

        if not args.sampling:
            optimizer.zero_grad()
            out = model(dataset_ind)
            if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
                if dataset_ind.y.shape[1] == 1:
                    # change -1 instances to 0 for one-hot transform
                    # dataset.label[dataset.label==-1] = 0
                    true_label = F.one_hot(dataset_ind.y, dataset_ind.y.max() + 1).squeeze(1)
                else:
                    true_label = dataset_ind.y

                loss = criterion(out[train_idx], true_label.squeeze(1)[
                                train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out[train_idx], dataset_ind.y.squeeze(1)[train_idx])
            loss.backward()
            optimizer.step()
        else:
            pbar = tqdm(total=train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')

            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(device) for adj in adjs]

                optimizer.zero_grad()
                out = model(dataset_ind, adjs, dataset_ind.x[n_id])
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, dataset_ind.y.squeeze(1)[n_id[:batch_size]])
                loss.backward()
                optimizer.step()
                pbar.update(batch_size)
            pbar.close()

        result = evaluate(model, dataset_ind, split_idx, eval_func, sampling=args.sampling, subgraph_loader=subgraph_loader)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_state = copy.deepcopy(model.state_dict())
            best_logit = result[-1]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred[dataset_ind.node_idx].unique(return_counts=True)[1].float()/pred[dataset_ind.node_idx].shape[0])
    logger.print_statistics(run)
    torch.save(best_state, f'{model_dir}/model{run}.pt')
    torch.save(best_logit, f'{model_dir}/logit{run}.pt')

### Save results ###
best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}-ind.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + 
                    f"{sub_dataset}" +
                    f"{best_val.mean():.2f} ± {best_val.std():.2f}," +
                    f"{best_test.mean():.2f} ± {best_test.std():.2f}\n")

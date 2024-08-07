import faulthandler
faulthandler.enable()

import os
import argparse
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger_detect
from data_utils import rand_splits, set_random_seed, evaluate_detect, eval_acc, eval_rocauc
from torch_geometric.utils import to_undirected
from dataset import load_dataset
from parse import parse_method, parser_add_main_args
from hyparams import hparams
from baselines import *


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
if args.dataset in hparams:
    for hname, v in hparams[args.dataset][args.method].items():
        setattr(args, hname, v)
print(args)

#fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args.dataset)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if len(dataset_ood_te.y.shape) == 1:
    dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

ood_type = args.ood_type

# infer the number of classes for non one-hot and one-hot labels
n = dataset_ind.num_nodes
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

if args.dataset not in ["arxiv-year", "snap-patents"] and dataset_ind.is_directed():
    dataset_ind.edge_index = to_undirected(dataset_ind.edge_index)

### Load method ###
model = OODGAT(d, c, False, True, True, args)
dataset_ind.x = dataset_ind.x.to(device)
dataset_ind.edge_index = dataset_ind.edge_index.to(device)
dataset_ind.y = dataset_ind.y.to(device)

if args.dataset in ('proteins', 'ppi'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### metric for classification ###
if args.dataset in ('proteins', 'ppi', 'twitch'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

### logger for result report ###
logger = Logger_detect(args.runs, args)
model_dir = f'checkpoints/{ood_type}/{args.dataset}/{args.method}' #{args.ood}
print(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# saved_indicator = '_'.join([f'{hname}-{v}' for hname, v in hparams[args.dataset][args.method].items()])

model.train()
print('MODEL:', model)

### Training loop ###
inference_time = []
durations = []
for run in range(args.runs):
    t = time.time()
    print(f'----start time: {t}')
    torch.manual_seed(run + args.seed)
    set_random_seed(run + args.seed)
    split_idx = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)
    dataset_ind.splits = split_idx

    model.reset_parameters()
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.ood == 'GPN':
        optimizer, _ = model.get_optimizer(lr=args.lr, weight_decay=args.weight_decay)
        warmup_optimizer = model.get_warmup_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('-inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        if args.ood == 'GPN' and epoch < args.GPN_warmup:
            warmup_optimizer.zero_grad()
            tick = time.time()
            loss = model.loss_compute(dataset_ind, device)
            loss.backward()
            warmup_optimizer.step()
        else:
            if args.ood == 'OODGAT': tick = time.time()
            optimizer.zero_grad()
            loss = model.loss_compute(dataset_ind, device, args, epoch)
            loss.backward()
            optimizer.step()
        
        result, score = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device, return_score=True)
        tock = time.time()
        inference_time.append(tock - tick)
        logger.add_result(run, result)

        if result[-1] > best_val:
            best_val = result[-1]
            best_state = copy.deepcopy(model.state_dict())
            # best_score = score

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss.detach().cpu().item():.4f}, '
                    f'AUROC: {100 * result[0]:.2f}%, '
                    f'AUPR: {100 * result[1]:.2f}%, '
                    f'FPR95: {100 * result[2]:.2f}%, '
                    f'Test Score: {100 * result[-2]:.2f}%')

        torch.cuda.empty_cache()
    
    # logger.print_statistics(run)
    # if best_state is None:
    #     best_state = copy.deepcopy(model.state_dict())
    #     best_score = score
    # torch.save(best_state, f'{model_dir}/model{run}.pt')
    # torch.save(best_score, f'{model_dir}/score{run}.pt')
    durations.append(time.time()-t)
    
print(f'======={args.ood}, time = {np.array(durations).mean():.5f}==========')
result = logger.print_statistics()
### Save results ###
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    if args.grasp:
        ood_name = f'{args.ood}+GRASP {args.st.title()} 50%'
    elif args.prop:
        ood_name = f'{args.ood}+prop'
    else:
        ood_name = f'{args.ood}+{args.OODGAT_detect_type}'
    # auroc, aupr, fpr, id ACC, train time
    write_obj.write(f"{sub_dataset}" + f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}," +
                    f"{result[:, -1].mean():.2f} ± {result[:, -1].std():.2f}," +
                    f"{np.mean(inference_time):.4f}\n")
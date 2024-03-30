import faulthandler
faulthandler.enable()

import os
import argparse
import random
import time
import copy
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger_detect
from data_utils import evaluate_detect, eval_acc, eval_rocauc
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
if args.ood_type=='label' and args.dataset in hparams:
    for hname, v in hparams[args.dataset][args.method].items():
        setattr(args, hname, v)
print(args)

#fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args.dataset, args.ood_type)

if len(dataset_ind.y.shape) == 1:
    dataset_ind.y = dataset_ind.y.unsqueeze(1)
if isinstance(dataset_ood_te, list):
    for data in dataset_ood_te:
        if len(data.y.shape) == 1:
            data.y = data.y.unsqueeze(1)
else:
    if len(dataset_ood_te.y.shape) == 1:
        dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

# get the splits for all runs
split_idx_lst = torch.load(f'./data/splits/{args.ood_type}/{args.dataset}-splits.pt')

# infer the number of classes for non one-hot and one-hot labels
n = dataset_ind.num_nodes
c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
d = dataset_ind.x.shape[1]

print(f"ind dataset {args.dataset}: all nodes {dataset_ind.num_nodes} | centered nodes {dataset_ind.node_idx.shape[0]} | edges {dataset_ind.edge_index.size(1)} | "
      + f"classes {c} | feats {d}")
if isinstance(dataset_ood_te, list):
    for i, data in enumerate(dataset_ood_te):
        print(f"ood te dataset {i} {args.dataset}: all nodes {data.num_nodes} | centered nodes {data.node_idx.shape[0]} | edges {data.edge_index.size(1)}")
else:
    print(f"ood te dataset {args.dataset}: all nodes {dataset_ood_te.num_nodes} | centered nodes {dataset_ood_te.node_idx.shape[0]} | edges {dataset_ood_te.edge_index.size(1)}")

if args.dataset not in ["arxiv-year", "snap-patents"] and dataset_ind.is_directed():
    dataset_ind.edge_index = to_undirected(dataset_ind.edge_index)

### Load method ###
if args.ood == 'GPN':
    model = GPN(d, c, args).to(device)
elif args.ood == 'SGCN':
    teacher = parse_method(args, dataset_ind, n, c, d, device)
    model = SGCN(d, c, args, dataset_ind).to(device)

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
model_dir = f'checkpoints/{args.ood_type}/{args.dataset}/{args.method}' #{args.ood}
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
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    dataset_ind.splits = split_idx

    model.reset_parameters()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.ood == 'GPN':
        optimizer, _ = model.get_optimizer(lr=args.lr, weight_decay=args.weight_decay)
        warmup_optimizer = model.get_warmup_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.ood == 'SGCN':
        tick = time.time()
        if args.gkde_use_pretrained:
            checkpoint = glob.glob(f'{model_dir}/model{run}*.pt')[0]
            teacher.load_state_dict(torch.load(checkpoint, map_location=device))
            teacher.eval()
        else:
            teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            for epoch in range(args.epochs):
                teacher.train()
                # for d_in, d_out in zip(train_loader_ind, train_loader_ood):
                teacher_optimizer.zero_grad()
                out = teacher(dataset_ind)
                out = F.log_softmax(out, dim=1)
                teacher_loss = criterion(
                    out[train_idx], dataset_ind.y.squeeze(1)[train_idx])
                teacher_loss.backward()
                teacher_optimizer.step()
        model.create_storage(dataset_ind, teacher, device)

    best_loss = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        if args.ood == 'GPN' and epoch < args.GPN_warmup:
            warmup_optimizer.zero_grad()
            tick = time.time()
            loss = model.loss_compute(dataset_ind)
            loss.backward()
            warmup_optimizer.step()
        else:
            optimizer.zero_grad()
            loss = model.loss_compute(dataset_ind)
            loss.backward()
            optimizer.step()
        
        result, score = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device, return_score=True)
        tock = time.time()
        inference_time.append(tock - tick)
        logger.add_result(run, result)

        if result[-1] < best_loss:
            best_loss = result[-1]
            best_state = copy.deepcopy(model.state_dict())
            # best_score = score

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'AUROC: {100 * result[0]:.2f}%, '
                    f'AUPR: {100 * result[1]:.2f}%, '
                    f'FPR95: {100 * result[2]:.2f}%, '
                    f'Test Score: {100 * result[-2]:.2f}%')

        torch.cuda.empty_cache()
    
    logger.print_statistics(run)
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        # best_score = score
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
        ood_name = f'{args.ood}+{args.GPN_detect_type}'
    ood_name += f'+{args.ood_type}'
    # auroc, aupr, fpr, id ACC, train time
    write_obj.write(f"{sub_dataset}" + f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}," +
                    f"{result[:, -1].mean():.2f} ± {result[:, -1].std():.2f}," +
                    f"{np.mean(inference_time):.4f}\n")
#!/bin/bash

device=0

for dataset in "cora" "amazon-photo" "coauthor-cs" "chameleon" "squirrel" "arxiv-year" "snap-patents" "wiki" "ogbn-products" "reddit2";
do
    python test_ood.py --dataset $dataset --device $device  --ood "MSP" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "Energy" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "KNN" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "ODIN" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "Mahalanobis" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "GNNSafe" --runs 5 
    python test_ood.py --dataset $dataset --device $device  --ood "GRASP" --runs 5 
done
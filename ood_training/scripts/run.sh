#!/bin/bash

device=0

for dataset in "cora" "amazon-photo" "coauthor-cs" "chameleon" "squirrel" "arxiv-year" "snap-patents" "wiki" "ogbn-products" "reddit2";
do
    # GKDE
    python train_ood.py --dataset cora --method gcn --ood SGCN  --GPN_detect_type Alea --device $device --runs 5
    python train_ood.py --dataset cora --method gcn --ood SGCN  --GPN_detect_type Epist --device $device --runs 5
    # GPN
    python train_ood.py --dataset cora --ood GPN --GPN_detect_type Alea --device $device --runs 5
    python train_ood.py --dataset cora --ood GPN --GPN_detect_type Epist --device $device --runs 5
    python train_ood.py --dataset cora --ood GPN --GPN_detect_type Epist_wo_Net --device $device --runs 5
    # OODGAT
    python train_oodgat.py --dataset cora --ood OODGAT  --OODGAT_detect_type ATT --device $device --runs 5
    python train_oodgat.py --dataset cora --ood OODGAT  --OODGAT_detect_type ENT --device $device --runs 5 
done
# GRASP: GRaph-Augmented Score Propagation for OOD detection
This is the official implementation of NeurIPS'24 paper [Revisiting Score Propagation in Graph Out-of-Distribution Detection](https://openreview.net/forum?id=jb5qN3212b).


## Dependence

- Ubuntu 20.04.6
- Cuda 11.3
- Pytorch 1.13.1
- Pytorch Geometric 2.3.1

## Dataset Preparation

- All small-scale datasets are already in the codebase or will be downloaded automatically during data loading.
- We upload all large-scale datasets except wiki to google drive [folder](https://drive.google.com/drive/folders/1gtLkgLMgSz9xrO8GG0rzUzyhrL60npX4?usp=sharing). Download these files and put these datasets to `data` foler.

- For dataset wiki, please download from the [link](https://www.kaggle.com/datasets/baimaxishi/large-scale-heterophily-graph-dataset-of-grasp). Then put the downloaded files to the `./data/` directory.

## Usage

### 1. Post-hoc Methods

All pretrained model checkpoints are already in the codebase. 

Run `./scripts/test_ood.sh` to evaluate the performance of all post-hoc methods. 

  
### 2. Training-based Methods

The folder `ood_training` contains all codes for training-based methods `GKDE`, `GPN` and `OODGAT`. Enter folder `ood_training` and run `scripts/run.sh` to evaluate these methods.

## Citation

Please cite our paper if you find it helpful.
```
@inproceedings{ma2024grasp,
 title = {Revisiting Score Propagation in Graph Out-of-Distribution Detection},
 author = {Ma, Longfei and Sun, Yiyou and Ding, Kaize and Liu, Zemin and Wu, Fei},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {4341--4373},
 volume = {37},
 year = {2024}
}
```

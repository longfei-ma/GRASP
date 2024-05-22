# GRASP: GRaph-Augmented Score Propagation for OOD detection
This is the source code for NeurIPS 2024 submitted paper [Revisiting Score Propagation in Graph Out-of-Distribution Detection](https://openreview.net/forum?id=jb5qN3212b).

## Dependence

- Ubuntu 20.04.6
- Cuda 11.3
- Pytorch 1.13.1
- Pytorch Geometric 2.3.1

## Dataset Preparation

- All small-scale datasets are already in the codebase or will be downloaded automatically during data loading.
- We upload all large-scale datasets except wiki and related splits file to google drive [folder](https://drive.google.com/drive/folders/1gtLkgLMgSz9xrO8GG0rzUzyhrL60npX4?usp=sharing). Download these files, put splits files to `./data/splits/semantic` folder and put data files to `./data/` foler.

- For dataset wiki, please download [wiki_features.pt](https://drive.google.com/file/d/1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK/view?usp=sharing), [wiki_edges.pt](https://drive.google.com/file/d/14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u/view?usp=sharing) from the links. Then put the downloaded files to the `./data/` directory.

## Usage

### 1. Post-hoc Methods

All pretrained model checkpoints are already in the codebase. 

Run `./scripts/test_ood.sh` to evaluate the performance of all post-hoc methods. 

  
### 2. Training-based Methods

The folder `ood_training` contains all codes for training-based methods `GKDE`, `GPN` and `OODGAT`. Enter folder `ood_training` and run `scripts/run.sh` to evaluate these methods.

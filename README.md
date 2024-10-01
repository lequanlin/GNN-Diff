# GNN-Diff: Boost Graph Neural Networks with Minimal Hyperparameter Tuning

## Description
This is the code for the paper "Diffusing to the Top: Boost Graph Neural Networks with Minimal Hyperparameter Tuning".

GNN-Diff is a graph-conditioned diffusion framework that generates high-performing parameters for a target GNN to match or even surpass the results of time-consuming hyperparameter tuning with a much more efficient process.

Please "star" us if you find the code helpful ✧◝(⁰▿⁰)◜✧

## Method Overview

How GNN-Diff works for node classification. (1) Input graph data: input graph signals, adjacency matrix, and train/validation ground truth labels. (2) Parameter collection: we use a coarse search with a small search space to select an appropriate hyperparameter configuration for the target GNN, and then collect model checkpoints with this configuration. (3) Training: PAE and GAE are firstly trained to produce latent parameters and the graph condition, and then G-LDM learns how to recover latent parameters from white noises with the graph condition as a guidance. (4) Inference: after sampling latent parameters from G-LDM, GNN parameters are reconstructed with the PAE decoder and returned to the target GNN for prediction.

![Method_v1](https://github.com/user-attachments/assets/b0b96f63-45e0-4977-9583-45ab979d2e35)

## Requirements
```bash 
hydra-core==1.3.2
matplotlib==3.7.3
numpy==1.24.4
pandas == 1.5.3
omegaconf==2.3.0
pytorch_lightning==2.1.2
scikit_learn==1.3.1
timm==0.4.12
tqdm==4.66.1
torch-sparse==0.6.17
torch-scatter==2.1.1
torch-geometric==2.5.3
ogb==1.3.6
```

## GNN Hyperparameter Tuning
```bash
gnn_tuning_nc.py  # Basic node classification
gnn_tuning_nc_large.py # Node classification on large graphs
gnn_tuning_nc_lr.py  # Node classification on long-range graphs
gnn_tuning_lp.py  # Link prediction
```

## GNN-Diff Training and Testing
```bash
train_gnn_diff.py
```

## Note

Our code is adapted from the implementation code of [[p-diff]](https://github.com/NUS-HPC-AI-Lab/Neural-Network-Parameter-Diffusion).



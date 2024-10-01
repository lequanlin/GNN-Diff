# GNN-Diff: Boost Graph Neural Networks with Minimal Hyperparameter Tuning

## Description
This is the code for the paper "Diffusing to the Top: Boost Graph Neural Networks with Minimal Hyperparameter Tuning".

GNN-Diff is a graph-conditioned diffusion framework that generates high-performing parameters for a target GNN to match or even surpass the results of time-consuming hyperparameter tuning with a much more efficient process.

## Method Overview

How GNN-Diff works for node classification. (1) Input graph data: input graph signals, adjacency matrix, and train/validation ground truth labels. (2) Parameter collection: we use a coarse search with a small search space to select an appropriate hyperparameter configuration for the target GNN, and then collect model checkpoints with this configuration. (3) Training: PAE and GAE are firstly trained to produce latent parameters and the graph condition, and then G-LDM learns how to recover latent parameters from white noises with the graph condition as a guidance. (4) Inference: after sampling latent parameters from G-LDM, GNN parameters are reconstructed with the PAE decoder and returned to the target GNN for prediction.

![Method_v1](https://github.com/user-attachments/assets/b0b96f63-45e0-4977-9583-45ab979d2e35)


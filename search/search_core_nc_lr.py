# -*- coding: utf-8 -*-
"""

@author: Lequan Lin

This is the code to support searching algorithms for GNNs on node classification.

"""

import numpy as np
import torch
from torch import nn
import torch_sparse as thsp
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Batch
from sklearn.metrics import f1_score
import argparse
import time
import os
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader, Sampler, Subset
from collections import defaultdict
import random
# import dgl
# from dgl.data import *
import os.path as osp
import pandas as pd
# from data.data_proc import load_data
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def weighted_cross_entropy(pred, true):
    """
    Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    num_nodes = true.size(0)
    num_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(num_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (num_nodes - cluster_sizes).float() / num_nodes
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight)
    # binary: in case we only have two unique classes in y
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
    return loss



### Define training and testing
def train(model, optimizer, data, args):
    model.train()
    if args.model.lower() in ['gcn', 'appnp', 'gin','sage', 'sgc', 'gpr', 'mixhop', 'mlp']:
        outputs = model(data.x, data.edge_index)

    # criterion = nn.CrossEntropyLoss()
    loss = weighted_cross_entropy(outputs, data.y)
    # loss = criterion(outputs, data.y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def eval_f1(model, data, args):
    model.eval()
    if args.model.lower() in ['gcn', 'appnp', 'gin','sage', 'sgc', 'gpr', 'mixhop', 'mlp']:
        logits = model(data.x, data.edge_index)

    pred = logits.max(1)[1].cpu().numpy()
    labels = data.y.cpu().numpy()

    # Calculate F1-score
    f1 = f1_score(labels, pred, average='macro', zero_division=0)

    return f1

# Collate function in training loader: we ensure that subgraphs are handled in a large graph properly
# The `Batch.from_data_list` function combines a list of `Data` objects into a single `Batch`
def custom_collate_fn(batch):
    return Batch.from_data_list(batch)

# We will apply channel-wise normalization to data.x
def channel_wise_normalization(data, mean, std):
    data.x = (data.x - mean) / std
    return data


### Define a function to load graph data
def load_data_lrgb(args, root='data', rand_seed=2023):
    dataset = args.dataset
    path = osp.join(root, dataset)
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset_train = torch.load(osp.join(path, 'dataset_train.pt'))
        dataset_val = torch.load(osp.join(path, 'dataset_val.pt'))
        dataset_test = torch.load(osp.join(path, 'dataset_test.pt'))

        train_loader = dataset_train['loader']
        val_loader = dataset_val['loader']
        test_loader = dataset_test['loader']

        num_features = dataset_train['num_features']
        num_classes = dataset_train['num_classes']

    except FileNotFoundError:
        if dataset in ['pascalvoc-sp']:
            dataset_train = LRGBDataset(path, dataset, split = 'train')
            dataset_val = LRGBDataset(path, dataset, split='val')
            dataset_test = LRGBDataset(path, dataset, split='test')
        elif dataset in ['coco-sp']:
            dataset_train = LRGBDataset(path, dataset, split='train')
            dataset_val = LRGBDataset(path, dataset, split='val')
            dataset_test = LRGBDataset(path, dataset, split='test')
            dataset_train = dataset_train[:11329]
            dataset_val = dataset_val[:500]
            dataset_test = dataset_test[:500]
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Find num_features and num_classes
        sample_data = dataset_train[0]
        num_features = sample_data.x.size(1)
        num_classes = dataset_train.num_classes

        ### Preprocess training data
        # Compute mean and std for channel-wise normalization
        x_all = torch.cat([data.x for data in dataset_train], dim=0)
        mean = x_all.mean(dim=0)
        std = x_all.std(dim=0)

        # Apply normalization to the training data
        dataset_train = [channel_wise_normalization(data, mean, std) for data in dataset_train]


        train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
        torch.save(dict(loader = train_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_train.pt'))

        ### Preprocessing validation and test data

        # Apply normalization to the validation and test data
        dataset_val = [channel_wise_normalization(data, mean, std) for data in dataset_val]
        dataset_test = [channel_wise_normalization(data, mean, std) for data in dataset_test]

        val_loader = DataLoader(dataset_val, batch_size = 500, shuffle = False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(dataset_test, batch_size = 500, shuffle = False, collate_fn=custom_collate_fn)

        torch.save(dict(loader=val_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_val.pt'))
        torch.save(dict(loader=test_loader, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset_test.pt'))

    return train_loader, val_loader, test_loader, num_features, num_classes


### Tunable hyperparameters and some basic settings
def get_args(model, dataset, search_method, optimizer, lr, num_hid = 16, num_layers = 2, dropout = 0.5, wd = 5e-2,
             appnp_alpha = 0.5, appnp_K = 2, sgck = 1, gpr_alpha = 0.1, gpr_K = 2):
    name_list = [('ExpNum', 'int')]
    parser = argparse.ArgumentParser()

    ## Basic settings

    parser.add_argument('--dataset', type=str, default='coco-sp', help='dataset names')
    name_list.append(('dataset', 'str'))

    parser.add_argument('--model', type=str, default='gcn2', help='GNN model names')
    name_list.append(('model', 'str'))

    parser.add_argument('--search_method', type=str, default='grid', help='Searching algorithm')
    name_list.append(('search_method', 'str'))

    parser.add_argument('--runs', type=int, default=3, help='Number of iterations for each hyper config')
    name_list.append(('runs', 'int'))

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    name_list.append(('epochs', 'int'))

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    name_list.append(('seed', 'int'))

    ## Hyperparameters for training

    parser.add_argument('--optimizer', type=str, default='Adam', choices = ['Adam','SGD'], help='The type of optimizer')
    name_list.append(('optimizer', 'str'))

    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    name_list.append(('lr', 'float'))
    
    parser.add_argument('--wd', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    name_list.append(('wd', 'float'))

    parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
    name_list.append(('dropout', 'float'))

    ## Hyperparameters for specific model architectures

    parser.add_argument('--num_hid', type=int, default=16, help='Number of hidden units.')
    name_list.append(('num_hid', 'int'))

    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers.')
    name_list.append(('num_layers', 'int'))

    parser.add_argument('--appnp_alpha', type=float, default=0.5, help='Teleport alpha in APPNP')
    name_list.append(('appnp_alpha', 'float'))

    parser.add_argument('--appnp_K', type=int, default=2, help='K in APPNP')
    name_list.append(('appnp_K', 'float'))

    parser.add_argument('--sgck', type=int, default=1, help='Degree of convolution in SGC')
    name_list.append(('sgck', 'int'))

    parser.add_argument('--gpr_alpha', type=float, default=0.1, help='Alpha in GPRGNN')
    name_list.append(('gpr_alpha', 'float'))

    parser.add_argument('--gpr_K', type=int, default=2, help='K in GPRGNN')
    name_list.append(('gpr_K', 'float'))

    args = parser.parse_args()
    args.model, args.dataset, args.search_method, args.optimizer, args.lr, args.num_hid, args.num_layers, args.dropout, args.wd  = \
        model, dataset, search_method, optimizer, lr, num_hid, num_layers, dropout, wd

    args.appnp_alpha, args.appnp_K, args.sgck, args.gpr_alpha, args.gpr_K = \
        appnp_alpha, appnp_K, sgck, gpr_alpha, gpr_K
   
    return args, name_list

def main(args, name_list):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, num_features, num_classes = load_data_lrgb(args)

    results_test = []
    results_val = []

    optimizers = {'AdamW': torch.optim.AdamW ,'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

    for run in range(args.runs):

        ## Get the model
        if args.model.lower() in ['gcn']:
            model = GCN_lr(num_features, num_classes, args.num_hid, num_layers = args.num_layers, dropout=args.dropout)
        elif args.model.lower() in ['appnp']:
            model = APPNP_net_lr(num_features, num_classes, num_hid = args. num_hid, K = args.appnp_K, alpha = args.appnp_alpha, dropout= args.dropout)
        elif args.model.lower() in ['sage']:
            model = SAGE_lr(num_features, num_classes, num_hid = args.num_hid, num_layers = args.num_layers, dropout=args.dropout)
        elif args.model.lower() in ['sgc']:
            model = SGC_lr(num_features, num_classes, num_hid = args.num_hid, K = args.sgck, dropout=args.dropout)
        elif args.model.lower() in ['gpr']:
            model = GPRGNN_lr(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout, alpha = args.gpr_alpha, K = args.gpr_K)
        elif args.model.lower() in ['mixhop']:
            model = MixHop_lr(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout, powers=[6, 8, 10])
        elif args.model.lower() in ['mlp']:
            model = MLP_lr(num_features, num_classes, num_hid = args.num_hid, num_layers = args.num_layers, dropout=args.dropout)


        model = model.to(device)

        optimizer = optimizers[args.optimizer](params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val_f1 = test_f1 = 0
        for epoch in range(1, args.epochs + 1):

            train_f1 = val_f1 = current_test_f1 = 0

            # Training
            for i, batch_train in enumerate(train_loader):
                batch_train = batch_train.to(device)
                batch_train.edge_index = to_undirected(batch_train.edge_index, num_nodes=batch_train.x.shape[0])
                train(model, optimizer, batch_train, args) #We train with 50 batches in each epoch
                with torch.no_grad():
                    train_f1_temp = eval_f1(model, batch_train, args)
                    train_f1 += train_f1_temp
                if i == 49: break
            train_f1 /= 50

            # Validation and testing
            with torch.no_grad():
                for i, batch_val in enumerate(val_loader):
                    batch_val = batch_val.to(device)
                    batch_val.edge_index = to_undirected(batch_val.edge_index, num_nodes=batch_val.x.shape[0])
                    val_f1_temp = eval_f1(model, batch_val, args)
                    val_f1 += val_f1_temp
                    if i == 2: break
                val_f1 /= 3
                # val_f1 /= len(val_loader)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

                with torch.no_grad():
                    for i, batch_test in enumerate(test_loader):
                        batch_test = batch_test.to(device)
                        batch_test.edge_index = to_undirected(batch_test.edge_index, num_nodes=batch_test.x.shape[0])
                        test_f1_temp = eval_f1(model, batch_test, args)
                        current_test_f1 += test_f1_temp
                    current_test_f1 /= len(test_loader)
                    test_f1 = current_test_f1

            log = 'Epoch: {:03d}, Train over epoch: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_f1, best_val_f1, test_f1))

        results_test.append(test_f1)
        results_val.append(best_val_f1)

        log = 'Iteration {} completes.'
        print(log.format(run + 1))


    results_test = torch.Tensor(results_test)
    results_val = torch.Tensor(results_val)
    print(f'Averaged val f1: {results_val.mean():.4f} \pm {results_val.std():.4f}.')
    print(f'Averaged test f1: {results_test.mean():.4f} \pm {results_test.std():.4f}.')
        
    ### Save the results
    output_dir = 'outputs/search'
    os.makedirs(output_dir, exist_ok=True)
    csv_name = args.dataset + '_' + args.model + '_' + args.search_method + '.csv'
    ResultCSV = os.path.join(output_dir, csv_name)

    if os.path.isfile(ResultCSV):
        df = pd.read_csv(ResultCSV)
        ExpNum = df['ExpNum'].iloc[-1] + 1
    else:
        outputs_names = {'ExpNum': 'int'}
        ExpNum = 1
        outputs_names.update({name: value for (name, value) in name_list})
        outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1,args.runs+1)})
        outputs_names.update({'Ave_Val_F1': 'float', 'Val_F1_std': 'float'})
        outputs_names.update({'Ave_Test_F1': 'float', 'Test_F1_std': 'float'})
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    new_row = {'ExpNum': ExpNum}
    new_row.update({name: value for (name, value) in args._get_kwargs()})
    new_row.update({'Replicate{0:2d}'.format(ii): results_test[ii-1] for ii in range(1,args.runs+1)})
    new_row.update({'Ave_Val_F1': results_val.mean().item(), 'Val_F1_std': results_val.std().item()})
    new_row.update({'Ave_Test_F1': results_test.mean().item(), 'Test_F1_std': results_test.std().item()})
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ResultCSV, index=False)


if __name__ == '__main__':
    main(*get_args(model, dataset, search_method, optimizer, lr ,num_hid, dropout, wd))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            




























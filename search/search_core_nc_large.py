# -*- coding: utf-8 -*-
"""

@author: Lequan Lin

This is the code to support searching algorithms for GNNs on node classification for large graphs.

"""

import numpy as np
import torch
from torch import nn
import torch_sparse as thsp
import torch.nn.functional as F
from torch_geometric.datasets import Reddit, Flickr
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import ClusterData, ClusterLoader
import argparse
import time
import os
import os.path as osp
import pandas as pd
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


### Define training and testing
def train(model, optimizer, data, train_mask, args):
    model.train()
    if args.model.lower() in ['gcn', 'appnp', 'sage', 'gpr']:
        outputs = model(data.x, data.edge_index)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def eval(model, data, mask, args):
    model.eval()
    if args.model.lower() in ['gcn', 'appnp', 'sage', 'gpr']:
        logits, accs = model(data.x, data.edge_index), []

    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

    return acc

def load_data_pyg(args, root='data', rand_seed=2023):
    dataset = args.dataset
    path = osp.join(root, dataset)
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['reddit']:
            dataset = Reddit(path)
        elif dataset in ['flickr']:
            dataset = Flickr(path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        data = dataset[0]
        num_features = data.x.size(1)
        num_classes = dataset.num_classes

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])

    return data, num_features, num_classes

def load_data_ogb(args, root='data', rand_seed=2023):
    dataset = args.dataset.lower()
    path = osp.join(root, dataset)
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset.pt'))
        data = dataset['data']
        num_features = dataset['num_features']
        num_classes = dataset['num_classes']

    except FileNotFoundError:
        if dataset in ['ogbn-arxiv', 'ogbn-products']:
            dataset = PygNodePropPredDataset(name=dataset, transform=T.ToSparseTensor())
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


        data = dataset[0]
        num_features = data.x.shape[1]
        num_classes = dataset.num_classes

        # Reshape y
        data.y = data.y.squeeze()

        # Generate train, val, test masks
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()
    # Convert SparseTensor to edge_index
    edge_index = data.adj_t.coo()  # Get COO format
    data.edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    # Ensure the edge_index is in LongTensor format
    data.edge_index = data.edge_index.long()

    return data, num_features, num_classes

### Tunable hyperparameters and some basic settings
def get_args(model, dataset, search_method, optimizer, lr, num_hid = 16, dropout = 0.5, wd = 5e-2,
             appnp_alpha = 0.5, gpr_alpha = 0.1):
    name_list = [('ExpNum', 'int')]
    parser = argparse.ArgumentParser()

    ## Basic settings

    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='dataset names')
    name_list.append(('dataset', 'str'))

    parser.add_argument('--model', type=str, default='gcn', help='GNN model names')
    name_list.append(('model', 'str'))

    parser.add_argument('--search_method', type=str, default='grid', help='Searching algorithm')
    name_list.append(('search_method', 'str'))

    parser.add_argument('--runs', type=int, default=3, help='Number of iterations for each hyper config')
    name_list.append(('runs', 'int'))

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    name_list.append(('epochs', 'int'))

    parser.add_argument('--batch_size', type=int, default=32, help='Batch_size')
    name_list.append(('batch_size', 'int'))

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

    parser.add_argument('--num_hid', type=int, default=32, help='Number of hidden units.')
    name_list.append(('num_hid', 'int'))

    parser.add_argument('--appnp_alpha', type=float, default=0.5, help='Teleport alpha in APPNP')
    name_list.append(('appnp_alpha', 'float'))

    parser.add_argument('--gpr_alpha', type=float, default=0.1, help='Alpha in GPRGNN')
    name_list.append(('gpr_alpha', 'float'))

    args = parser.parse_args()
    args.model, args.dataset, args.search_method, args.optimizer, args.lr, args.num_hid, args.dropout, args.wd  = \
        model, dataset, search_method, optimizer, lr, num_hid, dropout, wd

    args.appnp_alpha, args.gpr_alpha = appnp_alpha, gpr_alpha
   
    return args, name_list

def main(args, name_list):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset.lower() in ['ogbn-products']:
        data, num_features, num_classes = load_data_ogb(args)
        cluster_num_parts = 960
    elif args.dataset.lower() in ['ogbn-arxiv']:
        data, num_features, num_classes = load_data_ogb(args)
        cluster_num_parts = 32
    elif args.dataset.lower() in ['flickr']:
        data, num_features, num_classes = load_data_pyg(args)
        cluster_num_parts = 32
    elif args.dataset.lower() in ['reddit']:
        data, num_features, num_classes = load_data_pyg(args)
        if args.model.lower() in ['appnp']:
            cluster_num_parts = 960
        else:
            cluster_num_parts = 960

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    results_test = []
    results_val = []

    optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

    for run in range(args.runs):

        ## Get the model
        if args.model.lower() in ['gcn']:
            model = GCN2(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['appnp']:
            model = APPNP_net(num_features, num_classes, K = 2, alpha = args.appnp_alpha, dropout= args.dropout)
        elif args.model.lower() in ['sage']:
            model = SAGE2(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['gpr']:
            model = GPRGNN(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout, alpha = args.gpr_alpha)


        model = model.to(device)

        # Create clusters from the large graph
        cluster_data = ClusterData(data, num_parts=cluster_num_parts, recursive=False)
        # Create data loaders for the clusters: a train_loader for training and an eval_loader for valication and testing
        train_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        eval_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=False, num_workers=0)


        optimizer = optimizers[args.optimizer](params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 170], gamma = 0.2)

        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs + 1):

            train_acc = val_acc = current_test_acc = 0

            ### Training ###
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                train(model, optimizer, batch, batch.train_mask, args)
                with torch.no_grad():
                    train_acc_temp = eval(model, batch, batch.train_mask, args)
                    train_acc += train_acc_temp

            num_batchs = len(train_loader)
            train_acc = train_acc / num_batchs


            ### Validation and Testing ###
            for i, batch in enumerate(eval_loader):
                batch = batch.to(device)
                with torch.no_grad():
                    val_acc_temp = eval(model, batch, batch.val_mask, args)
                    test_acc_temp = eval(model, batch, batch.test_mask, args)
                    val_acc += val_acc_temp
                    current_test_acc += test_acc_temp

            num_batchs = len(eval_loader)
            val_acc = val_acc / num_batchs
            current_test_acc = current_test_acc / num_batchs

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = current_test_acc
            log = 'Epoch: {:03d}, Train over epoch: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))

            scheduler.step()

        results_test.append(test_acc)
        results_val.append(best_val_acc)

        log = 'Iteration {} completes.'
        print(log.format(run + 1))


    results_test = 100 * torch.Tensor(results_test)
    results_val = 100 * torch.Tensor(results_val)
    print(f'Averaged val accuracy: {results_val.mean():.2f} \pm {results_val.std():.2f}.')
    print(f'Averaged test accuracy: {results_test.mean():.2f} \pm {results_test.std():.2f}.')
        
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
        outputs_names.update({'Ave_Val_Acc': 'float', 'Val_Acc_std': 'float'})
        outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    new_row = {'ExpNum': ExpNum}
    new_row.update({name: value for (name, value) in args._get_kwargs()})
    new_row.update({'Replicate{0:2d}'.format(ii): results_test[ii-1] for ii in range(1,args.runs+1)})
    new_row.update({'Ave_Val_Acc': results_val.mean().item(), 'Val_Acc_std': results_val.std().item()})
    new_row.update({'Ave_Test_Acc': results_test.mean().item(), 'Test_Acc_std': results_test.std().item()})
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ResultCSV, index=False)


if __name__ == '__main__':
    main(*get_args(model, dataset, search_method, optimizer, lr ,num_hid, dropout, wd))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            




























# -*- coding: utf-8 -*-
"""

@author: Lequan Lin

This is the code to support searching algorithms for GNNs on link prediction.

Designed based on https://github.com/tomonori-masui/graph-neural-networks/blob/main/gnn_pyg_implementations.ipynb

"""

import numpy as np
import torch
from torch import nn
import torch_sparse as thsp
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, HeterophilousGraphDataset
import argparse
import time
import os
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
import os.path as osp
import pandas as pd
from models import *
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


### Define training and testing
def train(model, optimizer, train_data, args):
    model.train()
    if args.model.lower() in ['gcn', 'chebnet','appnp', 'sage', 'mlp']:
        node_embeddings = model(train_data.x, train_data.edge_index)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # sampling training negatives for every training epoch
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
        )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

    # outputs are the product of a pair of nodes on each edge
    outputs = (node_embeddings[edge_label_index[0]] * node_embeddings[edge_label_index[1]]).sum(dim=-1).view(-1)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, edge_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    pred = (outputs > 0.5).cpu().numpy()
    train_acc = np.mean(pred == edge_label.cpu().numpy())

    return train_acc

def eval(model, data, args):
    model.eval()
    if args.model.lower() in ['gcn', 'chebnet','appnp', 'sage', 'mlp']:
        node_embeddings = model(data.x, data.edge_index)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    outputs = (node_embeddings[data.edge_label_index[0]] * node_embeddings[data.edge_label_index[1]]).sum(dim=-1).view(-1).sigmoid()
    pred = (outputs > 0.5).cpu().numpy()
    acc = np.mean(pred == data.edge_label.cpu().numpy())
    return acc

### Define a function to load graph data and split the data for link prediction
### Our experiment settings follow https://github.com/tkipf/gae
def load_data_link(args, root='data', rand_seed=2023):
    dataset = args.dataset
    path = osp.join(root, dataset)
    dataset = dataset.lower()
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    try:
        dataset = torch.load(osp.join(path, 'dataset_link.pt'))
        data = dataset['data']
        num_features = dataset['num_features']

    except FileNotFoundError:
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        elif dataset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dataset)
        elif dataset in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(path, dataset)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        num_features = dataset.num_features
        data = dataset[0]

        del data.train_mask
        del data.val_mask
        del data.test_mask

        split = T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0,
        )
        data.train_link_data, data.val_link_data, data.test_link_data = split(data)

        torch.save(dict(data=data, num_features=num_features),
                   osp.join(path, 'dataset_link.pt'))

    return data, num_features


### Tunable hyperparameters and some basic settings
def get_args(model, dataset, search_method, optimizer, lr, num_hid = 16, dropout = 0.5, wd = 5e-2, chebk = 2, appnp_alpha = 0.5,
             ):
    name_list = [('ExpNum', 'int')]
    parser = argparse.ArgumentParser()

    ## Basic settings

    parser.add_argument('--dataset', type=str, default='cora', help='dataset names')
    name_list.append(('dataset', 'str'))

    parser.add_argument('--model', type=str, default='gcn2', help='GNN model names')
    name_list.append(('model', 'str'))

    parser.add_argument('--search_method', type=str, default='grid', help='Searching algorithm')
    name_list.append(('search_method', 'str'))

    parser.add_argument('--runs', type=int, default=10, help='Number of iterations for each hyper config')
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

    parser.add_argument('--chebk', type=int, default=2, help='Polynomial degree in ChebNet')
    name_list.append(('chebk', 'int'))

    parser.add_argument('--appnp_alpha', type=float, default=0.5, help='Teleport alpha in APPNP')
    name_list.append(('appnp_alpha', 'float'))


    args = parser.parse_args()
    args.model, args.dataset, args.search_method, args.optimizer, args.lr, args.num_hid, args.dropout, args.wd  = \
        model, dataset, search_method, optimizer, lr, num_hid, dropout, wd

    args.chebk, args.appnp_alpha = chebk, appnp_alpha
   
    return args, name_list

def main(args, name_list):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data, num_features = load_data_link(args)
    train_data = data.train_link_data.cuda()
    val_data = data.val_link_data.cuda()
    test_data = data.test_link_data.cuda()

    results_test = []
    results_val = []

    optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

    for run in range(args.runs):

        ## Get the model
        if args.model.lower() in ['gcn']:
            model = GCN2(num_features, args.num_hid, num_hid=args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['chebnet']:
            model = Cheb2(num_features, args.num_hid, K = args.chebk, num_hid=args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['appnp']:
            model = APPNP_net_link(num_features, args.num_hid, K = 2, alpha = args.appnp_alpha, dropout= args.dropout)
        elif args.model.lower() in ['sage']:
            model = SAGE2(num_features, args.num_hid, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['mlp']:
            model = MLP2(num_features, args.num_hid, num_hid = args.num_hid, dropout=args.dropout)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
        model = model.to(device)

        optimizer = optimizers[args.optimizer](params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 125, 150, 175], gamma = 0.2)

        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs + 1):

            train_acc = train(model, optimizer, train_data, args)
            with torch.no_grad():
                # train_acc = eval(model, train_data, args)
                val_acc = eval(model, val_data, args)
                tmp_test_acc = eval(model, test_data, args)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
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
    csv_name = 'link_' + args.dataset + '_' + args.model + '_' + args.search_method + '.csv'
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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            




























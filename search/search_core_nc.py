# -*- coding: utf-8 -*-
"""

This is the code to support searching algorithms for GNNs on node classification.

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
# import dgl
# from dgl.data import *
import os.path as osp
import pandas as pd
# from data.data_proc import load_data
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


### Define training and testing
def train(model, optimizer, data, train_mask, args):
    model.train()
    if args.model.lower() in ['gcn', 'gat','chebnet','appnp',
                              'gin','sage', 'sgc', 'gpr', 'mixhop', 'mlp']:
        outputs = model(data.x, data.edge_index)
    elif args.model.lower() in ['h2gcn']:
        outputs = model(data.x, data.adj_sparse)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test(model, data, train_mask, val_mask, test_mask, args):
    model.eval()
    if args.model.lower() in ['gcn', 'gat','chebnet','appnp',
                              'gin','sage', 'sgc', 'gpr', 'mixhop', 'mlp']:
        logits, accs = model(data.x, data.edge_index), []
    elif args.model.lower() in ['h2gcn']:
        logits, accs = model(data.x, data.adj_sparse), []

    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

### Define a function to load graph data
def load_data(args, root='data', rand_seed=2023):
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
        if dataset in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, dataset)
        elif dataset == 'cora_ml':
            dataset = CitationFull(path, dataset)
        elif dataset in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(path, dataset)
        elif dataset == 'actor':
            dataset = Actor(path)
        elif dataset in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(path, dataset)
        elif dataset in ['computers', 'photo']:
            dataset = Amazon(path, dataset)
        elif dataset in ['cs', 'physics']:
            dataset = Coauthor(path, dataset)
        elif dataset in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            dataset = HeterophilousGraphDataset(path, dataset)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]

        torch.save(dict(data=data, num_features=num_features, num_classes=num_classes),
                   osp.join(path, 'dataset.pt'))

    data.edge_index = to_undirected(data.edge_index, num_nodes =data.x.shape[0])

    return data, num_features, num_classes

### Define a function to generate train/val/test masks when the original datasets does not have
def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask


### Tunable hyperparameters and some basic settings
def get_args(model, dataset, search_method, optimizer, lr, num_hid = 16, dropout = 0.5, wd = 5e-2, chebk = 2, appnp_alpha = 0.5,
             h2k = 1, gat_heads = 4, sgck = 1, gpr_alpha = 0.1):
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

    parser.add_argument('--h2k', type=int, default=2, help='Number of layers in H2GCN')
    name_list.append(('h2k', 'int'))

    parser.add_argument('--gat_heads', type=int, default=2, help='Number of layers in H2GCN')
    name_list.append(('gat_heads', 'int'))

    parser.add_argument('--sgck', type=int, default=1, help='Degree of convolution in SGC')
    name_list.append(('sgck', 'int'))

    parser.add_argument('--gpr_alpha', type=float, default=0.1, help='Alpha in GPRGNN')
    name_list.append(('gpr_alpha', 'float'))

    args = parser.parse_args()
    args.model, args.dataset, args.search_method, args.optimizer, args.lr, args.num_hid, args.dropout, args.wd  = \
        model, dataset, search_method, optimizer, lr, num_hid, dropout, wd

    args.chebk, args.appnp_alpha, args.h2k, args.gat_heads, args.sgck, args.gpr_alpha = \
        chebk, appnp_alpha, h2k, gat_heads, sgck, gpr_alpha
   
    return args, name_list

def main(args, name_list):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data, num_features, num_classes = load_data(args)
    data = data.to(device)

    results_test = []
    results_val = []

    optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}

    for run in range(args.runs):

        if args.dataset.lower() in ['texas', 'cornell', 'wisconsin', 'actor','chameleon']:
            # Use only the first split
            train_mask = data.train_mask[:, 0]
            val_mask = data.val_mask[:, 0]
            test_mask = data.test_mask[:, 0]
        elif args.dataset.lower() in ['computers', 'photo', 'cs', 'physics']:
            train_mask, val_mask, test_mask = generate_split(data, num_classes)
        elif args.dataset.lower() in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
            train_mask, val_mask, test_mask = data.stores[0]['train_mask'][:, 0], data.stores[0]['val_mask'][:, 0], data.stores[0]['test_mask'][:, 0]
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
           

        ## Get the model
        if args.model.lower() in ['gcn']:
            model = GCN2(num_features, num_classes, args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['chebnet']:
            model = Cheb2(num_features, num_classes, K = args.chebk, num_hid=args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['appnp']:
            model = APPNP_net(num_features, num_classes, K = 2, alpha = args.appnp_alpha, dropout= args.dropout)
        elif args.model.lower() in ['h2gcn']:
            model = H2GCN(num_features, num_classes, k = args.h2k, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['sage']:
            model = SAGE2(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['gat']:
            model = GAT2(num_features, num_classes, heads = args.gat_heads, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['sgc']:
            # model = SGC2(num_features, num_classes, num_hid = args.num_hid, K = args.sgck, dropout=args.dropout)
            model = SGC(num_features, num_classes, K=args.sgck)
        elif args.model.lower() in ['gpr']:
            model = GPRGNN(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout, alpha = args.gpr_alpha)
        elif args.model.lower() in ['mixhop']:
            model = MixHop(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout)
        elif args.model.lower() in ['mlp']:
            model = MLP2(num_features, num_classes, num_hid = args.num_hid, dropout=args.dropout)

        data.adj_sparse = torch.sparse.FloatTensor(data.edge_index.to(device), torch.ones(data.edge_index.shape[1]).to(device),
                                                (data.x.shape[0], data.x.shape[0])).float().to(device)

        model = model.to(device)
        data = data.to(device)

        optimizer = optimizers[args.optimizer](params=model.parameters(), lr=args.lr, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 125, 150, 175], gamma = 0.2)

        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs + 1):

            train(model, optimizer, data, train_mask, args)
            with torch.no_grad():
                train_acc, val_acc, tmp_test_acc = test(model, data, train_mask, val_mask, test_mask, args)

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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            




























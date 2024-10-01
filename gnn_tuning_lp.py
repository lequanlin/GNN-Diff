   # -*- coding: utf-8 -*-
"""

Run this code for grid search, random search, and coarse search

Support GNNs for link prediction

Results are saved in folder outputs\search\link_dataset_model_method.csv

"""

from search.search_core_lp import *
import os
from itertools import product
from functools import reduce
from operator import mul
import numpy as np
from tqdm import tqdm
import time
import random

random.seed(42)

# Options for search_method: ['grid', 'coarse', 'random']
search_method = 'grid'

# Options for model: ['mlp', 'gcn', 'chebnet', 'appnp', 'sage']
model = 'sage'

# Options for dataset:
# ['cora', 'citeseer', 'pubmed', 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
dataset = ["citeseer"]

### Input your search space
optimizer = ['SGD', 'Adam']
lr= [0.005, 0.01, 0.05, 0.1, 0.5, 1]
num_hid =[16, 32, 64]
dropout = [0.1, 0.3, 0.5, 0.7, 0.9]
wd = [5e-4, 5e-3, 5e-2]

# Model specific hypers
chebk = [1, 2, 3] # ChebNet polynomial degree K
appnp_alpha = [0.1, 0.5, 0.9] # APPNP teleport alpha

time_list = []

if search_method in ['grid']:

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage', 'mixhop', 'mlp']:
            if model.lower() in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['mixhop']:
                print('### Search for MixHop starts')
            elif model.lower() in ['mlp']:
                print('### Search for MLP starts')
            else:
                print('### Search for SAGE starts')
            parameters = [optimizer, lr, num_hid, dropout, wd]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['chebnet']:
            print('### Search for ChebNet starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, chebk]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, chebk)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk= chebk_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, appnp_alpha]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha=appnp_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['h2gcn']:
            print('### Search for H2GCN starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, h2k]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, h2k)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k=h2k_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gat']:
            print('### Search for GAT starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, gat_heads]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, gat_heads)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads=gat_heads_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, sgck]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, sgck)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, sgck_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, sgck=sgck_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, gpr_alpha]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha=gpr_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

elif search_method in ['coarse']:

    num_configs = 54

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage', 'mixhop', 'mlp']:
            if model.lower() in ['gcn1', 'gcn2', 'gcn3']:
                print('### Search for GCN starts')
            elif model.lower() in ['mixhop']:
                print('### Search for MixHop starts')
            elif model.lower() in ['mlp']:
                print('### Search for MLP starts')
            else:
                print('### Search for SAGE starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['chebnet']:
            print('### Search for ChebNet starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, chebk)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk=chebk_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha=appnp_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['h2gcn']:
            print('### Search for H2GCN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, h2k)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k=h2k_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gat']:
            print('### Search for GAT starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gat_heads)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads=gat_heads_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            search_space = product(optimizer, lr, wd, sgck)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, wd_i, sgck_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer=optimizer_i, lr=lr_i, wd=wd_i, sgck=sgck_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha=gpr_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

elif search_method in ['random']:

    num_configs = 108

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage', 'mixhop', 'mlp']:
            if model.lower() in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['mixhop']:
                print('### Search for MixHop starts')
            elif model.lower() in ['mlp']:
                print('### Search for MLP starts')
            else:
                print('### Search for SAGE starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['chebnet']:
            print('### Search for ChebNet starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, chebk)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, chebk=chebk_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha=appnp_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['h2gnc']:
            print('### Search for H2GCN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, h2k)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, h2k=h2k_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gat']:
            print('### Search for GAT starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gat_heads)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gat_heads=gat_heads_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            search_space = product(optimizer, lr, wd, sgck)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, wd_i, sgck_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer=optimizer_i, lr=lr_i, wd=wd_i, sgck=sgck_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha=gpr_alpha_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

print('The {} search time of {} for dataset {} are {} (mins).'.format(search_method, model, dataset, time_list))





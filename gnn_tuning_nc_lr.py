   # -*- coding: utf-8 -*-
"""

Run this code for grid search, random search, and coarse search

Support GNNs for node classification on long range graph benchmarks

Results are saved in folder outputs\search\dataset_model_method.csv

"""

from search.search_core_nc_lr import *
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

# Options for model: ['mlp', 'gcn', 'appnp', 'sage', 'sgc', 'gpr', 'mixhop']
model = 'sage'

# Options for dataset:
# Long-range graphs ['pascalvoc-sp', 'coco-sp']
dataset = ['pascalvoc-sp']

### Input your search space
optimizer = ['AdamW']
lr= [0.0005, 0.001, 0.005]
num_hid =[192, 256]
num_layers = [8, 10]
dropout = [0.1, 0.3]
wd = [5e-4, 5e-3]

# Model specific hypers
appnp_alpha = [0.1, 0.5, 0.9] # APPNP teleport alpha
appnp_K = [8, 10] # APPNP K
sgck = [8, 10] # SGC K
gpr_alpha = [0.1, 0.5, 0.9] # GPRGNN alpha
gpr_K = [8, 10] # GPRGNN K


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
            parameters = [optimizer, lr, num_hid, num_layers, dropout, wd]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, num_layers, dropout, wd)
            for optimizer_i, lr_i, num_hid_i, num_layers_i, dropout_i, wd_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, num_layers=num_layers_i, dropout=dropout_i, wd=wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, appnp_alpha, appnp_K]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha, appnp_K)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i, appnp_K_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, appnp_alpha=appnp_alpha_i, appnp_K=appnp_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, sgck]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, sgck)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, sgck_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, wd=wd_i, sgck=sgck_i, dropout=dropout_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            parameters = [optimizer, lr, num_hid, dropout, wd, gpr_alpha, gpr_K]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha, gpr_K)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i, gpr_K_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, gpr_alpha=gpr_alpha_i, gpr_K=gpr_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

elif search_method in ['coarse']:

    num_configs = 10

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
            search_space = product(optimizer, lr, num_hid, num_layers, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, num_layers_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, num_layers=num_layers_i, dropout=dropout_i, wd=wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha, appnp_K)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i, appnp_K_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, appnp_alpha=appnp_alpha_i, appnp_K=appnp_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, sgck)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, sgck_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer=optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, sgck=sgck_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha, gpr_K)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i, gpr_K_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, gpr_alpha=gpr_alpha_i, gpr_K=gpr_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

elif search_method in ['random']:

    num_configs = 20

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage', 'mixhop', 'mlp']:
            if model.lower in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['mixhop']:
                print('### Search for MixHop starts')
            elif model.lower() in ['mlp']:
                print('### Search for MLP starts')
            else:
                print('### Search for SAGE starts')
            search_space = product(optimizer, lr, num_hid, num_layers, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, num_layers_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, num_layers_i, dropout_i, wd_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['appnp']:
            print('### Search for APPNP starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, appnp_alpha, appnp_K)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, appnp_alpha_i, appnp_K_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, appnp_alpha=appnp_alpha_i, appnp_K=appnp_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['sgc']:
            print('### Search for SGC starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, sgck)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, sgck_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer=optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, sgck=sgck_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        elif model.lower() in ['gpr']:
            print('### Search for GPRGNN starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd, gpr_alpha, gpr_K)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i, gpr_alpha_i, gpr_K_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr=lr_i, num_hid=num_hid_i, dropout=dropout_i, wd=wd_i, gpr_alpha=gpr_alpha_i, gpr_K=gpr_K_i))
                print('###Search for config {} is finished. Remain {} configs.'.format(iter, num_configs - iter))

        end_time = time.time()
        time_list.append((end_time - start_time) / 60)

print('The {} search time of {} for dataset {} are {} (mins).'.format(search_method, model, dataset, time_list))





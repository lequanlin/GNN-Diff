   # -*- coding: utf-8 -*-
"""

Run this code for grid search, random search, and coarse search

Support GNNs for node classification on largs graphs

Results are saved in folder outputs\search\dataset_model_method.csv

"""

from search.search_core_nc_large import *
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

# Options for model: ['gcn', 'appnp', 'sage', 'gpr']
model = 'gcn'

# Options for dataset:
# Large graphs ['ogbn-arxiv', 'ogbn-products', 'reddit', 'flickr']
dataset = ['reddit']

### Input your search space
optimizer = ['Adam']
lr= [0.0005, 0.005, 0.05]
num_hid =[64, 128, 256]
dropout = [0.1, 0.3, 0.7]
wd = [5e-4, 5e-3]

# Model specific hypers
appnp_alpha = [0.1, 0.3, 0.5, 0.9] # APPNP teleport alpha
gpr_alpha = [0.1, 0.2, 0.5, 0.9] # GPRGNN alpha


time_list = []

if search_method in ['grid']:

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage']:
            if model.lower() in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['sage']:
                print('### Search for SAGE starts')
            parameters = [optimizer, lr, num_hid, dropout, wd]
            num_configs = reduce(mul, [len(p) for p in parameters])
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in search_space:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
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

    num_configs = 10
    print('Total number of configs is', num_configs)

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage']:
            if model.lower() in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['sage']:
                print('### Search for SAGE starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
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

    num_configs = 20

    for data in dataset:

        start_time = time.time()
        iter = 0

        if model.lower() in ['gcn', 'sage']:
            if model.lower() in ['gcn']:
                print('### Search for GCN starts')
            elif model.lower() in ['sage']:
                print('### Search for SAGE starts')
            search_space = product(optimizer, lr, num_hid, dropout, wd)
            search_list = list(search_space)
            random_configs = random.sample(search_list, num_configs)
            for optimizer_i, lr_i, num_hid_i, dropout_i, wd_i in random_configs:
                print('###Current dataset is', data)
                iter += 1
                search = main(*get_args(model, data, search_method, optimizer_i, lr_i, num_hid_i, dropout_i, wd_i))
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





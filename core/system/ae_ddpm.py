import numpy as np
import random
import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM
from data.data_proc import load_data, generate_split, load_data_pyg_large, load_data_ogb
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.loader import ClusterData, ClusterLoader

import time

"""

Lena comment:

The model class is designed based on PyTorch Lightning 
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
A Lightning module typically includes:
- An initialization __init__
- A forward pass
- A training scheme: training_step()
- A validation scheme: validation_step()
- A test scheme: test_step()
- A prediction scheme: predict_step()
- An optimization and learning rate schedule setting: configure_optimizers()

"""
class AE_DDPM(DDPM):
    def __init__(self, config, **kwargs):

        param_ae_model =  hydra.utils.instantiate(config.system.param_ae_model)
        input_dim = config.system.param_ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = param_ae_model.encode(input_noise).shape
        config.system.model.arch.in_dim = latent_dim[-1] * latent_dim[-2] # The latent size of epsilon_theta input

        if config.task.data.dataset.lower() in ['reddit', 'flickr']:
            self.data, num_features, num_classes = load_data_pyg_large(config.task)
        elif config.task.data.dataset.lower() in ['ogbn-arxiv', 'ogbn-products']:
            self.data, num_features, num_classes = load_data_ogb(config.task)
        else:
            self.data, num_features, num_classes = load_data(config.task)


        self.gtask = config.system.graph_ae_model.task
        config.system.graph_ae_model.input_feat_dim = num_features
        config.system.graph_ae_model.output_dim = num_classes
        config.system.graph_ae_model.hidden_dim1 = latent_dim[-1] * latent_dim[-2]
        config.system.graph_ae_model.hidden_dim2 = latent_dim[-1] * latent_dim[-2]
        graph_ae_model = hydra.utils.instantiate(config.system.graph_ae_model)

        # model = hydra.utils.instantiate(config.system.model.arch.model)
        model = hydra.utils.instantiate(config.system.model.arch)

        ### Get data ###
        if config.task.data.dataset.lower() in ['texas', 'cornell', 'wisconsin', 'actor', 'chameleon']:
            # Use the first split to try
            self.train_mask = self.data.train_mask[:, 0]
            self.val_mask = self.data.val_mask[:, 0]
            self.test_mask = self.data.test_mask[:, 0]

        elif config.task.data.dataset.lower() in ['computers', 'photo', 'cs', 'physics']:
            self.train_mask, self.val_mask, self.test_mask = generate_split(self.data, num_classes)

        elif config.task.data.dataset.lower() in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers","questions"]:
            self.train_mask, self.val_mask, self.test_mask = self.data.stores[0]['train_mask'][:, 0], \
            self.data.stores[0]['val_mask'][:, 0], self.data.stores[0]['test_mask'][:, 0]

        elif config.task.data.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
            self.train_mask = self.data.train_mask
            self.val_mask = self.data.val_mask
            self.test_mask = self.data.test_mask

        elif config.task.data.dataset.lower() in ['reddit', 'flickr', 'ogbn-arxiv', 'ogbn-products']:
            if config.task.data.dataset.lower() in ['ogbn-products']:
                cluster_num_parts = 960
            elif config.task.data.dataset.lower() in ['ogbn-arxiv']:
                cluster_num_parts = 32
            elif config.task.data.dataset.lower() in ['reddit']:
                cluster_num_parts = 640
            elif config.task.data.dataset.lower() in ['flickr']:
                cluster_num_parts = 32
            # Create clusters from the large graph
            cluster_data = ClusterData(self.data, num_parts=cluster_num_parts, recursive=False)
            # Create a data loader for the clusters
            self.g_train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True, num_workers=0)
            self.g_eval_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=False, num_workers=0)



        if self.gtask in ['node_classification']:
            edge_index, edge_weight = gcn_norm(self.data.edge_index, num_nodes=self.data.x.shape[0],
                                               add_self_loops=True)
            self.data.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                     (self.data.x.shape[0], self.data.x.shape[0])).cuda()
            self.data.adj_gt = torch.sparse.FloatTensor(edge_index, torch.ones_like(edge_weight),
                                                        (self.data.x.shape[0], self.data.x.shape[0])).to_dense().cuda()

        super(AE_DDPM, self).__init__(config)

        self.save_hyperparameters()
        self.split_epoch_d = self.train_cfg.split_epoch_1
        self.split_epoch_p = self.train_cfg.split_epoch_2
        self.loss_func = nn.MSELoss()
        self.param_ae_model = param_ae_model
        self.graph_ae_model = graph_ae_model
        self.model = model

    def param_ae_forward(self, batch, **kwargs):
        output = self.param_ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    # GVAE with decoder for node classification
    def graph_ae_node_forward(self, **kwargs):
        data = self.data.cuda()
        output, _, _ = self.graph_ae_model(data.x, data.adj)
        loss = F.nll_loss(output[self.train_mask], data.y[self.train_mask])
        return loss

    # GVAE with decoder for node classification on large graphs
    def graph_ae_node_large_forward(self, data, **kwargs):
        data = data.cuda()
        output, _, _ = self.graph_ae_model(data.x, data.adj)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        return loss


    def training_step(self, batch, batch_idx, **kwargs):

        ddpm_optimizer, param_ae_optimizer, graph_ae_optimizer = self.optimizers()

        ### Train graph autoencoder for ddpm condition ###
        if self.current_epoch < self.split_epoch_d:

            if self.gtask == "node_classification":
                loss = self.graph_ae_node_forward(**kwargs)

                graph_ae_optimizer.zero_grad()
                self.manual_backward(loss)
                graph_ae_optimizer.step()

                self.validation_step(batch, batch_idx, **kwargs)

            elif self.gtask == "node_classification_large":
                for i, gbatch in enumerate(self.g_train_loader):
                    gbatch = gbatch.cuda()
                    edge_index, edge_weight = gcn_norm(gbatch.edge_index, num_nodes=gbatch.x.shape[0],
                                                       add_self_loops=True)
                    gbatch.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                          (gbatch.x.shape[0], gbatch.x.shape[0])).cuda()
                    loss = self.graph_ae_node_large_forward(gbatch, **kwargs)
                    graph_ae_optimizer.zero_grad()
                    self.manual_backward(loss)
                    graph_ae_optimizer.step()
                self.validation_step(batch, batch_idx, **kwargs)

                # graph_ae_optimizer.zero_grad()
                # self.manual_backward(loss)
                # graph_ae_optimizer.step()
                #
                # self.validation_step(batch, batch_idx, **kwargs)

        ### Before the split epoch, the autoencoder is trained ###
        elif  self.current_epoch < self.split_epoch_p: # current_epoch is a property of lightning, which grows with the training process

            loss = self.param_ae_forward(batch, **kwargs)
            param_ae_optimizer.zero_grad()
            self.manual_backward(loss)
            param_ae_optimizer.step()


        ### Train DDPM ###
        else:
            if self.gtask in ['node_classification']:
                data = self.data.cuda()
                _, graph_cond, _ = self.graph_ae_model.encode(data.x, data.adj)
            elif self.gtask in ['node_classification_large']:
                graph_cond = None
                for i, gbatch in enumerate(self.g_eval_loader):
                    gbatch = gbatch.cuda()
                    edge_index, edge_weight = gcn_norm(gbatch.edge_index, num_nodes=gbatch.x.shape[0],
                                                       add_self_loops=True)
                    gbatch.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                          (gbatch.x.shape[0], gbatch.x.shape[0])).cuda()
                    _, graph_cond_temp, _ = self.graph_ae_model.encode(gbatch.x, gbatch.adj)
                    graph_cond_temp = graph_cond_temp.mean(dim=0, keepdim=True)
                    if graph_cond is None:
                        graph_cond = torch.zeros(graph_cond_temp.shape, device='cuda')
                    graph_cond += graph_cond_temp
                graph_cond /= len(self.g_eval_loader)

            loss = self.forward(batch, graph_cond, **kwargs)
            ddpm_optimizer.zero_grad()
            self.manual_backward(loss)
            ddpm_optimizer.step()


        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        return {'loss': loss}

    ### Produce latent and use it as input of DDPM (see ddpm.py) ###
    def pre_process(self, batch):
        latent =  self.param_ae_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    ### After generation with DDPM, apply decoder to generated latent (see ddpm.py) ###
    def post_process(self, outputs):
        # pdb.set_trace()
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.param_ae_model.decode(outputs)

    def validation_step(self, batch, batch_idx, **kwargs: Any):

        if self.current_epoch < self.split_epoch_d:
            if self.gtask in ["node_classification"]:
                data = self.data.cuda()
                outputs, _, _ = self.graph_ae_model(data.x, data.adj)
                accs = []
                for mask in [self.train_mask, self.val_mask, self.test_mask]:
                    _, predicted = outputs[mask].max(1)
                    total = mask.sum().item()
                    correct = predicted.eq(data.y[mask]).sum().item()
                    acc = 100. * correct / total
                    accs.append(acc)
                print('Graph AE train accuracy {:.2f}, validation accuracy {:.2f}, test accuracy {:.2f}'.format(accs[0], accs[1],
                                                                                                    accs[2]))

                self.log('ae_acc', 0)
                self.log('best_g_acc', 0)

            elif self.gtask in ['node_classification_large']:
                for i, gbatch in enumerate(self.g_eval_loader):
                    gbatch = gbatch.cuda()
                    edge_index, edge_weight = gcn_norm(gbatch.edge_index, num_nodes=gbatch.x.shape[0],
                                                       add_self_loops=True)
                    gbatch.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                             (gbatch.x.shape[0], gbatch.x.shape[0])).cuda()
                    outputs, _, _ = self.graph_ae_model(gbatch.x, gbatch.adj)
                    accs = []
                    for mask in [gbatch.train_mask, gbatch.val_mask, gbatch.test_mask]:
                        _, predicted = outputs[mask].max(1)
                        total = mask.sum().item()
                        correct = predicted.eq(gbatch.y[mask]).sum().item()
                        acc = 100. * correct / total
                        accs.append(acc)
                    if i == 0: break
                print('Graph AE train accuracy {:.2f}, validation accuracy {:.2f}, test accuracy {:.2f}'.format(
                        accs[0], accs[1], accs[2]))
                self.log('ae_acc', 0)
                self.log('best_g_acc', 0)


        elif self.current_epoch < self.split_epoch_p:

            good_param = batch[:10]
            input_accs = []
            for i, param in enumerate(good_param):
                acc, test_loss, output_list = self.task_func_val(param)
                input_accs.append(acc)
            formatted_accs = ["{:.2f}".format(number) for number in input_accs]
            print("input model accuracy:{}".format(formatted_accs ))

            """
            AE reconstruction parameters
            """
            print('---------------------------------')
            print('Test the AE model')
            ae_rec_accs = []
            latent = self.param_ae_model.encode(good_param)
            print("latent shape:{}".format(latent.shape))
            ae_params = self.param_ae_model.decode(latent)
            print("ae params shape:{}".format(ae_params.shape))
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                acc, test_loss, output_list = self.task_func_val(param)
                ae_rec_accs.append(acc)

            best_ae = max(ae_rec_accs)
            formatted_accs = ["{:.2f}".format(number) for number in ae_rec_accs]
            print('AE reconstruction models accuracy:{}'.format(formatted_accs))
            print('AE reconstruction models best accuracy:{:.2f}'.format(best_ae))
            print('---------------------------------')
            # self.log('best_graph_acc', 100)
            self.log('ae_acc', best_ae)
            self.log('best_g_acc', 0)

        else:
            if self.gtask in ['node_classification']:
                data = self.data.cuda()
                _, graph_cond, _ = self.graph_ae_model.encode(data.x, data.adj)
            elif self.gtask in ['node_classification_large']:
                graph_cond = None
                for i, gbatch in enumerate(self.g_eval_loader):
                    gbatch = gbatch.cuda()
                    edge_index, edge_weight = gcn_norm(gbatch.edge_index, num_nodes=gbatch.x.shape[0],
                                                       add_self_loops=True)
                    gbatch.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                          (gbatch.x.shape[0], gbatch.x.shape[0])).cuda()
                    _, graph_cond_temp, _ = self.graph_ae_model.encode(gbatch.x, gbatch.adj)
                    graph_cond_temp = graph_cond_temp.mean(dim=0, keepdim=True)
                    if graph_cond is None:
                        graph_cond = torch.zeros(graph_cond_temp.shape, device='cuda')
                    graph_cond += graph_cond_temp
                graph_cond /= len(self.g_eval_loader)
            dict = super(AE_DDPM, self).validation_step(batch, batch_idx, graph_cond, **kwargs)
            self.log('ae_acc', 0)
            return dict

    def test_step(self, batch, batch_idx, **kwargs: Any):
        if self.gtask in ['node_classification']:
            data = self.data.cuda()
            _, graph_cond, _ = self.graph_ae_model.encode(data.x, data.adj)


        elif self.gtask in ['node_classification_large']:
            graph_cond = None
            for i, gbatch in enumerate(self.g_eval_loader):
                gbatch = gbatch.cuda()
                edge_index, edge_weight = gcn_norm(gbatch.edge_index, num_nodes=gbatch.x.shape[0],
                                                   add_self_loops=True)
                gbatch.adj = torch.sparse.FloatTensor(edge_index, edge_weight,
                                                      (gbatch.x.shape[0], gbatch.x.shape[0])).cuda()
                _, graph_cond_temp, _ = self.graph_ae_model.encode(gbatch.x, gbatch.adj)
                graph_cond_temp = graph_cond_temp.mean(dim=0, keepdim=True)
                if graph_cond is None:
                    graph_cond = torch.zeros(graph_cond_temp.shape, device='cuda')
                graph_cond += graph_cond_temp
            graph_cond /= len(self.g_eval_loader)

        super(AE_DDPM, self).test_step(batch, batch_idx, graph_cond, **kwargs)

    def configure_optimizers(self, **kwargs):
        ae_parmas = self.param_ae_model.parameters()
        ddpm_params = self.model.parameters()
        graph_ae_params = self.graph_ae_model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ddpm_params)
        self.ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ae_parmas)
        self.graph_ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, graph_ae_params)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer, self.ae_optimizer, self.graph_ae_optimizer



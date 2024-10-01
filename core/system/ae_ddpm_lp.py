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
from data.data_proc import load_data_link
from torch_geometric.utils import negative_sampling

import time

"""

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
class AE_DDPM_LP(DDPM):
    def __init__(self, config, **kwargs):

        param_ae_model =  hydra.utils.instantiate(config.system.param_ae_model)
        input_dim = config.system.param_ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = param_ae_model.encode(input_noise).shape
        config.system.model.arch.in_dim = latent_dim[-1] * latent_dim[-2] # The latent size of epsilon_theta input

        if config.task.data.dataset.lower() in ['cora', 'citeseer', 'pubmed', 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']:
            self.data, num_features = load_data_link(config.task)
            self.train_data, self.val_data, self.test_data = self.data.train_link_data.cuda(), self.data.val_link_data.cuda(), self.data.test_link_data.cuda()
        else:
            raise ValueError(f"Unsupported dataset: {config.task.data.dataset}")

        self.gtask = config.system.graph_ae_model.task
        config.system.graph_ae_model.input_feat_dim = num_features
        config.system.graph_ae_model.hidden_dim1 = latent_dim[-1] * latent_dim[-2]
        config.system.graph_ae_model.hidden_dim2 = latent_dim[-1] * latent_dim[-2]
        graph_ae_model = hydra.utils.instantiate(config.system.graph_ae_model)

        model = hydra.utils.instantiate(config.system.model.arch)

        super(AE_DDPM_LP, self).__init__(config)

        self.save_hyperparameters()
        self.split_epoch_d = self.train_cfg.split_epoch_1
        self.split_epoch_p = self.train_cfg.split_epoch_2
        self.loss_func = nn.MSELoss()
        self.param_ae_model = param_ae_model
        self.graph_ae_model = graph_ae_model
        self.model = model

        self.graph_cond = None

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['graph_cond'] = self.graph_cond
    #
    # def on_load_checkpoint(self, checkpoint):
    #     self.graph_cond = checkpoint.get('graph_cond', None)

    def param_ae_forward(self, batch, **kwargs):
        output = self.param_ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    # GVAE with decoder for link prediction
    def graph_ae_link_forward(self, train_data, **kwargs):
        node_embeddings, _, _ = self.graph_ae_model(train_data.x, edge_index=train_data.edge_index)

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

        return loss


    def training_step(self, batch, batch_idx, **kwargs):

        ddpm_optimizer, param_ae_optimizer, graph_ae_optimizer = self.optimizers()

        ### Train graph autoencoder for ddpm condition ###
        if self.current_epoch < self.split_epoch_d:
            loss = self.graph_ae_link_forward(self.train_data, **kwargs)
            graph_ae_optimizer.zero_grad()
            self.manual_backward(loss)
            graph_ae_optimizer.step()
            self.validation_step(batch, batch_idx, **kwargs)


        ### Before the split epoch, the autoencoder is trained ###
        elif  self.current_epoch < self.split_epoch_p: # current_epoch is a property of lightning, which grows with the training process
            loss = self.param_ae_forward(batch, **kwargs)
            param_ae_optimizer.zero_grad()
            self.manual_backward(loss)
            param_ae_optimizer.step()

        ### Train DDPM ###
        else:
            graph_cond = self.graph_ae_model.encode_lp(self.train_data.x, self.train_data.edge_index)
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

            node_embeddings, _, _ = self.graph_ae_model(self.val_data.x, edge_index=self.val_data.edge_index)
            outputs = (node_embeddings[self.val_data.edge_label_index[0]] * node_embeddings[self.val_data.edge_label_index[1]]).sum(
                dim=-1).view(-1).sigmoid()
            pred = (outputs > 0.5).cpu().numpy()
            val_acc = np.mean(pred == self.val_data.edge_label.cpu().numpy())

            node_embeddings, _, _ = self.graph_ae_model(self.test_data.x, edge_index=self.test_data.edge_index)
            outputs = (node_embeddings[self.test_data.edge_label_index[0]] * node_embeddings[
                self.test_data.edge_label_index[1]]).sum(
                dim=-1).view(-1).sigmoid()
            pred = (outputs > 0.5).cpu().numpy()
            test_acc = np.mean(pred == self.test_data.edge_label.cpu().numpy())

            print('Graph AE validation acc {:.4f}, test acc {:,.4f}'.format(val_acc, test_acc))
            self.log('ae_acc', 0)
            self.log('best_g_acc', 0)


        elif self.current_epoch < self.split_epoch_p:

            good_param = batch[:10]
            input_accs = []
            for i, param in enumerate(good_param):
                acc, test_loss, output_list = self.task_func_val(param)
                input_accs.append(acc)
            formatted_accs = ["{:.4f}".format(number) for number in input_accs]
            print("input model acc:{}".format(formatted_accs ))

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
            formatted_accs = ["{:.4f}".format(number) for number in ae_rec_accs]
            print('AE reconstruction models acc:{}'.format(formatted_accs))
            print('AE reconstruction models best acc:{:.4f}'.format(best_ae))
            print('---------------------------------')
            self.log('ae_acc', best_ae)
            self.log('best_g_acc', 0)

        else:
            graph_cond = self.graph_ae_model.encode_lp(self.train_data.x, self.train_data.edge_index)
            dict = super(AE_DDPM_LP, self).validation_step(batch, batch_idx, graph_cond, **kwargs)
            self.log('ae_acc', 0)
            return dict

    def test_step(self, batch, batch_idx, **kwargs: Any):
        graph_cond = self.graph_ae_model.encode_lp(self.train_data.x, self.train_data.edge_index)
        super(AE_DDPM_LP, self).test_step(batch, batch_idx, graph_cond, **kwargs)

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



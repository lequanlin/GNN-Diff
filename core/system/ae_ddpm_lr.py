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
from .ddpm_lr import DDPM_LR
from data.data_proc import load_data_lrgb
from sklearn.metrics import f1_score

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
class AE_DDPM_LR(DDPM_LR):
    def __init__(self, config, **kwargs):

        param_ae_model =  hydra.utils.instantiate(config.system.param_ae_model)
        input_dim = config.system.param_ae_model.in_dim
        input_noise = torch.randn((1, input_dim))
        latent_dim = param_ae_model.encode(input_noise).shape
        config.system.model.arch.in_dim = latent_dim[-1] * latent_dim[-2] # The latent size of epsilon_theta input

        if config.task.data.dataset.lower() in ['pascalvoc-sp', 'coco-sp']:
            self.g_train_loader, self.g_val_loader, self.g_test_loader, num_features, num_classes \
                = load_data_lrgb(config.task)
        else:
            raise ValueError(f"Unsupported dataset: {config.task.data.dataset}")

        self.gtask = config.system.graph_ae_model.task
        config.system.graph_ae_model.input_feat_dim = num_features
        config.system.graph_ae_model.output_dim = num_classes
        config.system.graph_ae_model.hidden_dim1 = latent_dim[-1] * latent_dim[-2]
        config.system.graph_ae_model.hidden_dim2 = latent_dim[-1] * latent_dim[-2]
        graph_ae_model = hydra.utils.instantiate(config.system.graph_ae_model)

        model = hydra.utils.instantiate(config.system.model.arch)

        super(AE_DDPM_LR, self).__init__(config)

        self.save_hyperparameters()
        self.split_epoch_d = self.train_cfg.split_epoch_1
        self.split_epoch_p = self.train_cfg.split_epoch_2
        self.loss_func = nn.MSELoss()
        self.param_ae_model = param_ae_model
        self.graph_ae_model = graph_ae_model
        self.model = model

        self.graph_cond = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint['graph_cond'] = self.graph_cond

    def on_load_checkpoint(self, checkpoint):
        self.graph_cond = checkpoint.get('graph_cond', None)

    def weighted_cross_entropy(self, pred, true):
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

    def param_ae_forward(self, batch, **kwargs):
        output = self.param_ae_model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    # GVAE with decoder for node classification on long-range graphs
    def graph_ae_node_lr_forward(self, data, **kwargs):
        data = data.cuda()
        output, _, _ = self.graph_ae_model(data.x, edge_index = data.edge_index)
        loss = self.weighted_cross_entropy(output, data.y)
        return loss


    def training_step(self, batch, batch_idx, **kwargs):

        ddpm_optimizer, param_ae_optimizer, graph_ae_optimizer = self.optimizers()

        ### Train graph autoencoder for ddpm condition ###
        if self.current_epoch < self.split_epoch_d:

            for i, gbatch in enumerate(self.g_train_loader):
                gbatch = gbatch.cuda()
                loss = self.graph_ae_node_lr_forward(gbatch, **kwargs)
                graph_ae_optimizer.zero_grad()
                self.manual_backward(loss)
                graph_ae_optimizer.step()
                if i == 49: break

            if self.current_epoch == 0: print('Training of GAE starts.')
            if self.current_epoch == self.split_epoch_d - 1: print('Training of GAE finishes.')

            # self.validation_step(batch, batch_idx, **kwargs)

            if self.current_epoch == self.split_epoch_d - 1:
                graph_cond = None
                for i, gbatch in enumerate(self.g_train_loader):
                    gbatch = gbatch.cuda()
                    graph_cond_temp = self.graph_ae_model.encode_lr(gbatch.x, edge_index = gbatch.edge_index)
                    graph_cond_temp = graph_cond_temp.mean(dim=0, keepdim=True)
                    if graph_cond is None:
                        graph_cond = torch.zeros(graph_cond_temp.shape, device='cuda')
                    graph_cond += graph_cond_temp
                graph_cond /= len(self.g_train_loader)
                self.graph_cond = graph_cond.detach()

        ### Before the split epoch, the autoencoder is trained ###
        elif  self.current_epoch < self.split_epoch_p: # current_epoch is a property of lightning, which grows with the training process
            loss = self.param_ae_forward(batch, **kwargs)
            param_ae_optimizer.zero_grad()
            self.manual_backward(loss)
            param_ae_optimizer.step()

        ### Train DDPM ###
        else:
            loss = self.forward(batch, self.graph_cond, **kwargs)
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

            val_f1  = 0
            for i, gbatch in enumerate(self.g_val_loader):
                gbatch = gbatch.cuda()
                outputs, _, _ = self.graph_ae_model(gbatch.x, edge_index = gbatch.edge_index)
                pred = outputs.max(1)[1].cpu().numpy()
                labels = gbatch.y.cpu().numpy()
                # Calculate F1-score
                f1_temp = f1_score(labels, pred, average='macro', zero_division=0)
                val_f1 += f1_temp
            val_f1 /= len(self.g_val_loader)

            print('Graph AE validation f1 {:.4f}'.format(
                val_f1))
            self.log('ae_f1', 0)
            self.log('best_g_f1', 0)


        elif self.current_epoch < self.split_epoch_p:

            good_param = batch[:10]
            input_f1s = []
            for i, param in enumerate(good_param):
                f1, test_loss, output_list = self.task_func_val(param)
                input_f1s.append(f1)
            formatted_f1s = ["{:.4f}".format(number) for number in input_f1s]
            print("input model f1:{}".format(formatted_f1s ))

            """
            AE reconstruction parameters
            """
            print('---------------------------------')
            print('Test the AE model')
            ae_rec_f1s = []
            latent = self.param_ae_model.encode(good_param)
            print("latent shape:{}".format(latent.shape))
            ae_params = self.param_ae_model.decode(latent)
            print("ae params shape:{}".format(ae_params.shape))
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                f1, test_loss, output_list = self.task_func_val(param)
                ae_rec_f1s.append(f1)

            best_ae = max(ae_rec_f1s)
            formatted_f1s = ["{:.4f}".format(number) for number in ae_rec_f1s]
            print('AE reconstruction models f1:{}'.format(formatted_f1s))
            print('AE reconstruction models best f1:{:.4f}'.format(best_ae))
            print('---------------------------------')
            self.log('ae_f1', best_ae)
            self.log('best_g_f1', 0)

        else:
            dict = super(AE_DDPM_LR, self).validation_step(batch, batch_idx, self.graph_cond, **kwargs)
            self.log('ae_f1', 0)
            return dict

    def test_step(self, batch, batch_idx, **kwargs: Any):

        super(AE_DDPM_LR, self).test_step(batch, batch_idx, self.graph_cond.cuda(), **kwargs)

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



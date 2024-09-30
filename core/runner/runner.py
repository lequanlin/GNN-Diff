import pdb
import numpy as np
import random


import hydra.utils
import torch
import pytorch_lightning as pl
import sys
import os
import datetime
from core.tasks.node_classification import NCFTask
from core.system import *
import torch
import torch.distributed as dist
from core.tasks import tasks
import time

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

def set_seed(seed):
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('medium')
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg, **kwargs):
    cfg = cfg

    print("config:")
    for k, v in cfg.items():
        print(k, v)
    print("=" * 20)

    print("kwargs:")
    for k, v in kwargs.items():
        print(k, v)
    print("=" * 20)

    # set seed
    set_seed(cfg.seed)

    # set device
    set_device(cfg.device)

    # set process title
    set_processtitle(cfg)

def train_generation(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    start_time = time.time()
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.checkpoint_path)
    end_time = time.time()


    start_time_t = time.time()
    trainer.test(system, datamodule=datamodule)
    end_time_t = time.time()

    print('Training time cost (mins):', (end_time - start_time) / 60)
    print('Sampling time cost (mins):', (end_time_t - start_time_t) / 60)

    return {}

def test_generation(cfg):
    # init_experiment(cfg)
    # set device
    set_device(cfg.device)
    # set process title
    set_processtitle(cfg)

    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    trainer.test(system, datamodule=datamodule, ckpt_path=cfg.checkpoint_path)

    return {}

def train_task_for_data(cfg, **kwargs):
    init_experiment(cfg, **kwargs)
    task_cls = tasks[cfg.task.name]
    task = task_cls(cfg.task, **kwargs)

    task_result = task.train_for_data() # The core part of data preparation
    return task_result



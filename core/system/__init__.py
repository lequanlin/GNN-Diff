from .base import *
from .ddpm import *
from .ddpm_lr import *
from .encoder import *
from .vae import VAESystem
from .ae_ddpm import AE_DDPM
from .ae_ddpm_lr import AE_DDPM_LR
from .ae_ddpm_lp import AE_DDPM_LP

systems = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
    'ddpm_lr': DDPM_LR,
    'vae': VAESystem,
    'ae_ddpm': AE_DDPM,
    'ae_ddpm_lr': AE_DDPM_LR,
    'ae_ddpm_lp': AE_DDPM_LP,
}
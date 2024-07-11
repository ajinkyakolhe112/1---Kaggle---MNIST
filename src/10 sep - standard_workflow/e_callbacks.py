import torch
from lightning.pytorch.callbacks import Callback

import wandb
from lightning.pytorch.loggers import WandbLogger

wandb.login()

class CustomCallback(Callback):
    def __init__(self, track_this="epochs", verbose=True):
        self.track_this = track_this
        self.verbose = verbose
        self.state = {}
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass
    def on_train_epoch_end(self, trainer, pl_module):
        pass
        
        
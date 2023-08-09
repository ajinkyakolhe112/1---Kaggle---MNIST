import torch
import torch.nn as nn
import pytorch_lightning

class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        pass

class Extended(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        pass

    def train_step():
        pass

    def validation_step():
        pass

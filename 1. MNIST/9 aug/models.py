import torch
import torch.nn as nn
import pytorch_lightning

class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X

class Extended(pytorch_lightning.LightningModule):
    def __init__(self, init_model: torch.nn.Module):
        super().__init__()
        self.automatic_optimization = False
        self.model = init_model
    
    def forward(self, X):
        return self.model(X)

    def train_step(self, X, Y):
        optimizer = self.optimizers()
        
        Y_predicted = self.model(X)
        loss        = nn.functional.cross_entropy(Y_predicted, Y)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        torch.optim.Adam(self.model.parameters(), lr = 0.001)
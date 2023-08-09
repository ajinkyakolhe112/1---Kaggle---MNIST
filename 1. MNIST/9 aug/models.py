import torch
import torch.nn as nn
import pytorch_lightning

class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = nn.Linear(10,1)

    def forward(self, X):
        return X

class Extended(pytorch_lightning.LightningModule):
    def __init__(self, init_model: torch.nn.Module):
        super().__init__()
        self.automatic_optimization = False
        self.model = init_model
    
    def forward(self, X):
        return self.model(X)

    def training_step(self, batch_XY, batch_no):
        X, Y = batch_XY
        optimizer = self.optimizers()
        
        Y_predicted = self.model(X)
        loss        = nn.functional.cross_entropy(Y_predicted, Y)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        W_PARAMETERS = self.model.parameters()
        optimizer = torch.optim.Adam(W_PARAMETERS , lr = 0.001)
        return optimizer

def get_model():
    model = Baseline()
    extended = Extended(model)

    return extended
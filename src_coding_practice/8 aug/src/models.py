#%%
import torch
import torch.nn as nn
import skorch
from torch.nn import ReLU as RELU
from pytorch_lightning import LightningModule,Trainer
import pytorch_lightning

from dataloader import get_dataloaders
from loguru import logger


#%%
# Universal Approximate Function Approximator. 40 wide in layer 1. layer 2 for classification directly
class Baseline_Model(torch.nn.Module):
    def __init__(self, width= None):
        super().__init__()

        self.width = width

        self.sequential_model = nn.ModuleDict({
            "RESHAPE":  nn.Identity(),
            "layer1":   nn.Linear(28*28*1, self.width),
            "relu":     RELU(),
            "fc1":      nn.Linear(self.width, 10),
            "softmax":  nn.Softmax(dim=1),
        })

    def forward(self, independent_vars_batch):
        X_batch = independent_vars_batch
        X_batch = X_batch.view(-1,28*28*1)
        # logger.debug(f'B,Pixels Shape: {X_batch.shape}')
        
        tmp_output = self.sequential_model.layer1  (X_batch)
        tmp_output = self.sequential_model.relu    (tmp_output)
        tmp_output = self.sequential_model.fc1     (tmp_output)
        Y_predicted_probs = nn.functional.softmax(tmp_output, dim=1)
        # logger.debug(f'Sum of probs across batch = {Y_predicted_probs.sum(dim=1)}')

        return Y_predicted_probs

#%%
def test_single_example():
    test_img = torch.randn(3,1,28,28)
    container    = Baseline_Model(13)
    container(test_img)

#%%
class Lightning_Module(pytorch_lightning.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.automatic_optimization=False
        self.model = model
    
    def forward(self,x):
        return self.model(x)

    def training_step(self, x_y_batch):
        x_actual, y_actual = x_y_batch
        y_actual = y_actual.type(torch.float32)
        y_predicted_probs = self.model(x_actual)
        loss = nn.functional.cross_entropy(y_predicted_probs, y_actual)
        
        optimizer = self.optimizers()
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        self.log("train_loss",loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, x_actual, y_actual):
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


#%%
def get_lighning_model():
    pytorch_model   = Baseline_Model(13)
    lightning_model = Lightning_Module(pytorch_model)
    return lightning_model
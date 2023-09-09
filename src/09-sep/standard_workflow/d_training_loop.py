import torch
import lightning
import torchmetrics
import os

from loguru import logger

class NN_Training_Loop(lightning.LightningModule):
    def __init__(self, pytorch_model: torch.nn.Module):
        super().__init__()
        self.model = pytorch_model
        self.automatic_optimization= False
        
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

    def forward(self, X):
        self.model(X)

    def training_step(self, batch_XY, batch_no):
        X,Y       = batch_XY
        optimizer = self.optimizers()
        
        # 1. y_predicted = f(X, W)
        Y_predicted = self.model(X)
        
        # 2. loss = error_function(y_predicted, y_actual)
        loss        = torch.nn.functional.cross_entropy(Y, Y_predicted)
        
        # 3. dE/dW. Minimize E by gradient based minimization
        self.manual_backward(loss)
        
        # 4. W = W - dE/dW * learning_rate
        optimizer.step()
        
        # 5. Clean Up dE/dW at each neuron.
        optimizer.zero_grad()

        self.log("loss",loss.item(), prog_bar=True)
        self.accuracy(Y_predicted,Y)
        self.log('train_acc_step', self.accuracy, prog_bar=True)

    def validation_step(self, batch_XY, batch_no):
        X,Y       = batch_XY
        Y_predicted = self.model(X)
        loss        = torch.nn.functional.cross_entropy(Y, Y_predicted)

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.model.parameters(), lr= 0.01)


def test_lightning_module():
    from a_dataset import get_datasets
    from b_dataloader import get_dataloaders
    from c_model import Baseline_NN
    
    train_dataset, test_dataset = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset)
    
    model               = Baseline_NN()
    lightning_module    = NN_Training_Loop(model) 
    
    trainer_module      = lightning.Trainer(max_epochs = 5)
    trainer_module.fit( lightning_module, train_loader, val_loader  )


if __name__=="__main__":
    test_lightning_module()
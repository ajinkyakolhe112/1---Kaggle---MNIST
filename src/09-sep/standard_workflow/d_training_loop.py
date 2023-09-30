import torch
import lightning
import torchmetrics
import os
from loguru import logger

ACCURACY            = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
CONFUSION_MATRIX    = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=10)

class NN_Training_Loop(lightning.LightningModule):
    def __init__(self, pytorch_model: torch.nn.Module):
        super().__init__()
        self.model = pytorch_model
        self.automatic_optimization= False
        
        self.accuracy = ACCURACY
        self.confusion_matrix = CONFUSION_MATRIX
        
    def forward(self, x_actual):
        self.model(x_actual)

    def training_step(self, batch_data, batch_no):
        x_actual,y_actual = batch_data
        optimizer = self.optimizers()
        
        # 1. y_predicted = f(X, W)
        y_predicted = self.model(x_actual)
        # 2. loss = error_function(y_predicted, y_actual)
        loss        = torch.nn.functional.cross_entropy(y_actual, y_predicted)
        # 3. dE/dW. Minimize E by gradient based minimization
        self.manual_backward(loss)
        # 4. W = W - dE/dW * learning_rate
        optimizer.step()
        # 5. Clean Up dE/dW at each neuron.
        optimizer.zero_grad()

        monitor_train(self, x_actual, y_actual, y_predicted, loss)
    
    def monitor_train(self, x_actual, y_actual, y_predicted, loss):
        self.log("loss",loss.item(), prog_bar=True)
        
        batch_accuracy = self.accuracy(y_predicted,y_actual)
        self.log('train_acc_step', batch_accuracy, prog_bar=True)
        
    def validation_step(self, batch_data, batch_no):
        x_actual,y_actual = batch_data
        y_predicted = self.model(x_actual)
        loss        = torch.nn.functional.cross_entropy(y_actual, y_predicted)

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.model.parameters(), lr= 0.01)
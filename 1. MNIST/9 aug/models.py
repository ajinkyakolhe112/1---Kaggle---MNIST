import torch
import torch.nn as nn
import pytorch_lightning

class Baseline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28*28,15)
        self.class_predictor = nn.Linear(15, 10)

    def forward(self, X):
        output = self.layer_1(X)
        output = nn.functional.relu(output)

        output = self.class_predictor(output)
        output_probs = nn.functional.softmax(output, dim=1)

        return output_probs

class Extended(pytorch_lightning.LightningModule):
    def __init__(self, init_model: torch.nn.Module):
        super().__init__()
        self.automatic_optimization = False
        self.pytorch_model = init_model
    
    def forward(self, X):
        return self.pytorch_model(X)

    def training_step(self, batch_XY, batch_no):
        X, Y = batch_XY
        optimizer = self.optimizers()
        
        Y_predicted = self.pytorch_model(X)
        loss        = nn.functional.cross_entropy(Y_predicted, Y)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        W_PARAMETERS = self.pytorch_model.parameters()
        optimizer = torch.optim.Adam(W_PARAMETERS , lr = 0.001)
        return optimizer

def get_model():
    model = Baseline()
    extended = Extended(model)

    return extended
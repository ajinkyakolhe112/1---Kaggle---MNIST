import pytorch_lightning
import torch
import torchmetrics

from loguru import logger

class Extended(pytorch_lightning.LightningModule):
    def __init__(self, pytorch_model: torch.nn.Module):
        super().__init__()
        self.model = pytorch_model
        self.automatic_optimization= False

    def forward(self, X):
        self.model(X)

    def training_step(self, batch_XY, batch_no):
        X,Y       = batch_XY
        optimizer = self.optimizers()

        Y_predicted = self.model(X)
        loss        = torch.nn.functional.cross_entropy(Y_predicted, Y)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    def validation_step(self, batch_XY, batch_no):
        X,Y       = batch_XY
        Y_predicted = self.model(X)
        loss        = torch.nn.functional.cross_entropy(Y_predicted, Y)

    def configure_optimizers(self):
        return torch.optim.SGD(params = self.model.parameters(), lr= 0.001)

def get_lightning_module(pytorch_model=None):
    if pytorch_model is None:
        from models.baseline import Baseline_NN
        pytorch_model    = Baseline_NN()
    lightning_module = Extended(pytorch_model)

    return lightning_module

def test_lightning_module():
    from b_dataloader import get_dataloaders

    train_loader, val_loader, test_loader   = get_dataloaders()
    lightning_module                        = get_lightning_module()
    trainer_module = pytorch_lightning.Trainer(max_epochs = 5)
    trainer_module.fit( lightning_module, train_loader, val_loader  )

    logger.debug(f'')


if __name__=="__main__":
    test_lightning_module()
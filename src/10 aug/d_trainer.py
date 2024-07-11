import pytorch_lightning
import torch

def get_trainer():
    trainer = pytorch_lightning.Trainer(max_epochs=10)
    return trainer
import torch
import pytorch_lightning

class CustomCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        self.log()

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


def get_trainer():
    return trainer
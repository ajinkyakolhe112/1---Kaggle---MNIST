import pytorch_lightning
import torch
from pytorch_lightning.loggers import TensorBoardLogger


experiment_logger  = TensorBoardLogger(save_dir="lightning_logs", name="finding_overfitting_point", version="v2_")

"""
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        self.log()

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

callback_function = MyPrintingCallback()
"""
trainer = pytorch_lightning.Trainer(max_epochs=10, logger= experiment_logger, )

def get_trainer():
    return trainer
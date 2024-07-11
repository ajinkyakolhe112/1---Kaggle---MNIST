from a_dataset          import get_datasets
from b_dataloader       import get_dataloaders
from c_model            import Baseline_NN
from d_training_loop    import NN_Training_Loop
from e_callbacks        import CustomCallback  

train_dataset, test_dataset = get_datasets()
train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset)

model             = Baseline_NN()
lightning_module  = NN_Training_Loop(model)

from lightning.pytorch import Trainer

experiment = "end-to-end-pipeline"
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
experiment_logger  = TensorBoardLogger(save_dir="lightning_logs", name= experiment, version="v1_")

trainer = Trainer(max_epochs=5, logger= experiment_logger, callbacks = [CustomCallback()] )
trainer.fit( lightning_module, train_loader, val_loader  )
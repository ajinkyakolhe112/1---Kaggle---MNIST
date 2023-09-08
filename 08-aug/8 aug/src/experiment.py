#%%
from dataloader import get_dataloaders
from models import get_lighning_model

train_loader, val_loader, final_test_loader = get_dataloaders()
model   = get_lighning_model()

#%%
import pytorch_lightning

trainer = pytorch_lightning.Trainer()
trainer.fit(model, train_loader)

#%%
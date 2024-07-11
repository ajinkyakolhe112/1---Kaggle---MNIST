import torch
from loguru import logger

options_dict = {
    "batch_size": 5,
    "shuffle": True,
}

def get_dataloaders(train_dataset, test_dataset):
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **options_dict)
    val_loader   = torch.utils.data.DataLoader(val_dataset, **options_dict)
    test_loader  = torch.utils.data.DataLoader(test_dataset, **options_dict)

    return train_loader, val_loader, test_loader



if __name__=="__main__":
    test_dataloader()
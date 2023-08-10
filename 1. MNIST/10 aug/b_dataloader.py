import torch

from loguru import logger

def get_dataloaders(train_data, test_data):
    
    train_data, val_data = torch.utils.data.random_split(train_data, [0.9, 0.1])

    options_dict = {
        "batch_size": 32,
        "shuffle": True,
    }
    train_loader = torch.utils.data.DataLoader(train_data, **options_dict)
    val_loader   = torch.utils.data.DataLoader(val_data, **options_dict)
    test_loader  = torch.utils.data.DataLoader(test_data, **options_dict)

    return train_loader, val_loader, test_loader

def test_dataloader(data_loader):
    for batch_no, (image_tensors, label_vector) in enumerate(data_loader):
        logger.debug(f'image tensor shape = {image_tensors.shape}, label vector shape = {label_vector.shape}')
        if batch_no == 3:
            break

if __name__=="__main__":
    from a_datasets import get_datasets
    
    train_data, test_data = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_data, test_data)
    test_dataloader(train_loader)
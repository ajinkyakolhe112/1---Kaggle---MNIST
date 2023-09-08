import torch

from loguru import logger

def get_dataloaders(train_dataset, test_dataset):
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    options_dict = {
        "batch_size": 32,
        "shuffle": True,
    }
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **options_dict)
    val_loader   = torch.utils.data.DataLoader(val_dataset, **options_dict)
    test_loader  = torch.utils.data.DataLoader(test_dataset, **options_dict)

    return train_loader, val_loader, test_loader

def test_dataloader():
    from a_dataset import get_datasets
    
    train_data, test_data = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_data, test_data)

    for batch_no, (image_tensors, label_vector) in enumerate(train_loader):
        logger.debug(f'image tensor shape = {image_tensors.shape}, label vector shape = {label_vector.shape}')
        if batch_no == 3:
            break

if __name__=="__main__":
    test_dataloader()
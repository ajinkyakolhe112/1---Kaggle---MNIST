def test_dataloader():
    from a_dataset import get_datasets
    
    train_data, test_data = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_data, test_data)

    for batch_no, (image_tensors, label_vector) in enumerate(train_loader):
        logger.debug(f'current batch index {batch_no}')
        logger.debug(f'image tensor shape = {image_tensors.shape}, label vector shape = {label_vector.shape}')
        
        if batch_no == 3:
            logger.debug(f'reached batch index 3. exiting')
            break
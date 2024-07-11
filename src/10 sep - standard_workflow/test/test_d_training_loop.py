def test_lightning_module():
    from a_dataset import get_datasets
    from b_dataloader import get_dataloaders
    from c_model import Baseline_NN
    
    train_dataset, test_dataset = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, test_dataset)
    
    model               = Baseline_NN()
    lightning_module    = NN_Training_Loop(model) 
    
    trainer_module      = lightning.Trainer(max_epochs = 5)
    trainer_module.fit( lightning_module, train_loader, val_loader  )
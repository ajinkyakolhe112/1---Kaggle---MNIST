from loguru import logger
import wand as weights_biases
import pandas as pd # torch is not reinventing pandas. Hence need to use it for reading
import pkgutil


import torch # torch.utils.data
import torch.nn as nn

from tensorflow import keras
from tensorflow.keras import preprocessing, datasets, utils # equivalent places where data related functions live
from torchvision import datasets, transforms, ops as vision_ops, utils as vision_utils
"""
all modules in keras
1. datasets
2. preprocessing

4. layers, models
5. activations
6. intializers

6. losses, metrics
7. optimizers
8. regularizers

9. callbacks
10. utils
"""

# dir(torch) # lists all the functions in one place
# for i in pkgutil.iter_modules(torch.__path__):
#     print(i)
# dir(keras) # only has sub modules, not any functions. difference between pytorch & keras

def get_dataset_from_csv(path,train_flag):
    df = pd.read_csv(path)
    logger.debug(f'{df.info()}')
    logger.debug(f'Training Flag = {train_flag}')
    
    if train_flag:
        y = df.pop('label').to_numpy()
        y = torch.tensor(y)
        Y = nn.functional.one_hot(y,10)
        logger.debug(f'TESTING Y shape: {Y.shape}')

    X = df.to_numpy()
    X = torch.tensor(X)
    X = X.reshape(len(X),28,28,1)
    X = X/255.0 # values of X in range of 0 to 255. making it uniform 0 to 1
    logger.debug(f'TESTING X shape: {X.shape}')

    if train_flag:
        dataset = torch.utils.data.TensorDataset(X,Y) # TensorDataset takes tensors as input. Dataset is an abstract class
    else:
        dataset = torch.utils.data.TensorDataset(X)
    
    return dataset


def get_dataloaders():
    training_dataset   = get_dataset_from_csv("../input/digit-recognizer/train.csv", train_flag=True)
    final_test_dataset = get_dataset_from_csv("../input/digit-recognizer/test.csv" , train_flag=False)

    training_dataset, validation_dataset = torch.utils.data.dataset.random_split(training_dataset, [0.9, 0.1])

    training_dataloader   = torch.utils.data.DataLoader(training_dataset,      batch_size=32, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,    batch_size=32, shuffle=True) 
    final_test_dataloader = torch.utils.data.DataLoader(final_test_dataset, batch_size=32, shuffle=False) # maintain order for kaggle submission
    
    return training_dataloader, validation_dataloader, final_test_dataloader

def test_dataloader():
    train_loader, val_loader, final_test_loader = get_dataloaders()
    for batch_no, (image_tensors, labels_onehot) in enumerate(train_loader):
        print(batch_no)
        break

test_dataloader()
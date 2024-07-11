#%%
import pandas as pd
import numpy as np
import os
from loguru import logger

def get_final_test_dataset():
    final_test_df    = pd.read_csv(PATH+"test.csv")
    final_test_data = final_test_df.to_numpy()
    final_test_data = final_test_data.reshape(-1,28,28,1)
    
    return final_test_data

def get_training_dataset():
    PATH = "../input/digit-recognizer/"
    training_data_df = pd.read_csv(PATH+"train.csv")
    
    TARGET = training_data_df.label
    y = TARGET

    training_data_df.pop("label")
    X = training_data_df

    X,y = X.to_numpy(), y.to_numpy()

    X = X.reshape(-1,28,28,1) # B,H,W,C Channels Last
    X = X/255.0
    Y = np.zeros(shape=(len(y),10))
    i = 0
    for element in y:
        Y[i][element] = 1 # Only 1 index is value "1". Rest are all zero
        i = i+1

    logger.debug(f'X & Y shape = {X.shape}{Y.shape}')

    np.random.seed(5)
    indexes = np.random.choice(len(X),int(len(X)/2))
    X,Y = X[indexes],Y[indexes]

    return X,Y

#%%
import torch

def get_dataloaders():
    X,Y = get_training_dataset()
    X,Y = torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32)

    training_dataset = torch.utils.data.TensorDataset(X,Y)
    training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [0.9, 0.1])

    training_dataloader   = torch.utils.data.DataLoader(training_dataset,
                                batch_size=32, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                batch_size=32, shuffle=True)
    
    return training_dataloader, validation_dataloader

def test_everything():
    X,Y = get_training_dataset()
    train_loader, val_loader = get_dataloaders()

    logger.debug(f'')

test_everything()
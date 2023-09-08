import pandas as pd
import numpy as np
import torch
import os
from loguru import logger

print(os.getcwd())

def get_datasets():
    
    train_df = pd.read_csv("./input/digit-recognizer/train.csv")
    test_df  = pd.read_csv("./input/digit-recognizer/test.csv")

    logger.debug(f'{train_df.info()}')

    y = train_df['label']

    train_df.pop('label')
    X = train_df

    # Convert X in Pandas -> Numpy -> Torch Tensor
    X,y = X.to_numpy(), y.to_numpy()
    X,y = torch.tensor(X, dtype=torch.float32, requires_grad=False), torch.tensor(y, dtype=torch.float32, requires_grad=False)
    
    # X format = (Number of Examples, Number of Dimentions)
    # y format = ( Class Label Vector of Classes Dimention)
    # X.shape  = (42000,784)
    X = X.reshape(42000,28,28,1) # we are reshaping in order to allow the data in X.shape to be reformated to a format, readable by conv2d. which treats this data as an image.
    # Y.shape = (42000)
    Y = torch.nn.functional.one_hot(y.to(torch.int64), 10) # one hot expects, datatype of y to be torch.int64. one hot means, only one columns is hot / 1(relevant). rest all are 0. 
    Y = torch.Tensor.to(Y,dtype=torch.float32)
    # Y = torch.tensor(Y, dtype=torch.float32) 

    test_data_X = test_df.to_numpy()
    test_data_X = torch.tensor(test_data_X, dtype=torch.float32, requires_grad=False)
    test_data_X = test_data_X.reshape(-1,28,28,1)

    # Tensor Array to Tensor Dataset format.
    train_data = torch.utils.data.TensorDataset(X,Y) # dataloader always needs datatype = Dataset. Hence we are reformatting
    test_data  = torch.utils.data.TensorDataset(test_data_X)

    logger.debug(f'X_train[0] = {train_data[0][0].shape}, Y_train[0] = {train_data[0][1].shape}')
    logger.debug(f'X_test[0] = {test_data[0][0].shape}')

    return train_data, test_data



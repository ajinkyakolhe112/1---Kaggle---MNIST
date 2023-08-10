import pandas as pd
import numpy as np
import torch

from loguru import logger

def get_datasets():
    train_df = pd.read_csv("../input/digit-recognizer/train.csv")
    test_df  = pd.read_csv("../input/digit-recognizer/test.csv")

    logger.debug(f'{train_df.info()}')

    y = train_df['label']

    train_df.pop('label')
    X = train_df

    X,y = X.to_numpy(), y.to_numpy()
    X,y = torch.tensor(X, dtype=torch.float32, requires_grad=False), torch.tensor(y, dtype=torch.int64, requires_grad=False)
    
    X = X.reshape(-1,28,28,1)
    Y = torch.nn.functional.one_hot(y, 10) # one hot expects, datatype of y to be torch.int64.
    Y = torch.tensor(Y, dtype=torch.float32) 

    test_data_X = test_df.to_numpy()
    test_data_X = torch.tensor(test_data_X, dtype=torch.float32, requires_grad=False)
    test_data_X = test_data_X.reshape(-1,28,28,1)

    train_data = torch.utils.data.TensorDataset(X,Y)
    test_data  = torch.utils.data.TensorDataset(test_data_X)

    logger.debug(f'X_train[0] = {train_data[0][0].shape}, Y_train[0] = {train_data[0][1].shape}')
    logger.debug(f'X_test[0] = {test_data[0][0].shape}')

    return train_data, test_data
    pass

def test_datasets():
    get_datasets()

if __name__=="__main__":
    test_datasets()

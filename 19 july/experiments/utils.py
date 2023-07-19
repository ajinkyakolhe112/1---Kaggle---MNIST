# %% [code]
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
import pytorch_lightning as pl

from loguru import logger
from IPython.display import display,Latex
import wandb
# %% [markdown]
"""
Code Sections
1. Dataset
2. Model Architecture
3. Model Training
4. Experiment Monitoring 
"""
# %% [code]
data_path = "./input/"

train_data_df = pd.read_csv(data_path+"digit-recognizer/train.csv")
test_data_df = pd.read_csv(data_path+"digit-recognizer/test.csv")

logger.info(f'TODO: train_data shape needed x=(B,C,H,W), y=(B,10)')
y = train_data_df.label
x = train_data_df.drop("label", axis = 1) # axis 0 row, axis 1 column, axis 2 channels

x_client = test_data_df.to_numpy()
x_client = x_client.reshape(len(x_client),28,28,1)

from pandas import get_dummies                   # element to vector in pandas
from torch.nn.functional import one_hot          # element to vector in torch

y = pd.get_dummies(y) # numpy doesn't have one hot encoding. torch & pandas does.
assert y.shape[1] == 10
logger.debug(f'y reshape done. {y.shape}')

x = x.to_numpy()
y = y.to_numpy()

B,Pixels = x.shape
C = 1
H = int(np.sqrt(Pixels).item())
x = x.reshape(B,H,H,C)
assert len(x.shape) == 4
logger.debug(f'x reshape done. {x.shape}')
logger.info(f'{x.shape}{y.shape}')

logger.info(f'TypeError: Exception encountered when calling layer "conv2d" (type Conv2D)')
logger.info(f'Value passed to parameter `input` has DataType int64 not in list of allowed values: float16, bfloat16, float32, float64, int32')
x = np.array(x, dtype=np.float32)


#%% [code]
logger.info(f'TODO: train test split')
# pandas & numpy aren't made for train test split. they are made to just manage numeric arrays in python

from sklearn.model_selection import train_test_split     # easiest to use
# from torch.utils.data import random_split                # works as expected
# from tensorflow.data.Dataset import random_split         # deprecated, doc says use random
# from tensorflow.data.Dataset import random               # complex to use. couldn't figure out

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

logger.debug(f'train test split done')
logger.debug(f'{x_train.shape}{y_train.shape}\t{x_val.shape}{y_val.shape}')

#%%
def save_predictions_csv(y_predicted_probs):
    y_predicted_class = np.argmax(y_predicted_probs, axis=0)
    y_indexes = np.arange(0, len(y_predicted_class), 1)
    csv_file = open("predictions.csv","w")
    csv_file.write("ImageId,Label")
    np.savetxt(csv_file, np.concatenate((y_indexes, y_predicted_class)), delimiter=",")
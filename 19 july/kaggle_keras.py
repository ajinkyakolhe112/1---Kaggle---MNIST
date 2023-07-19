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
data_path = "../input/"

train_data_df = pd.read_csv(data_path+"digit-recognizer/train.csv")
test_data_df = pd.read_csv(data_path+"digit-recognizer/test.csv")

logger.info(f'TODO: train_data shape needed x=(B,C,H,W), y=(B,10)')
y = train_data_df.label
x = train_data_df.drop("label", axis = 1) # axis 0 row, axis 1 column, axis 2 channels

# added later. when predicting on final data. because it was needed only then
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

# %% [code]
logger.info(f'TODO: build NN archiecture')

from tensorflow.keras import models, layers, activations, losses, metrics, optimizers

feature_extractor = models.Sequential([
    layers.Conv2D(9,(7,7), activation="relu", input_shape = (28,28,1), data_format="channels_last"),     # layer is functioning as block
    layers.Conv2D(9,(7,7), activation="relu"),     # Calculating 7 as 28/4
    layers.Conv2D(9,(7,7), activation="relu"),
    layers.Conv2D(9,(7,7), activation="relu"),
])
keras_model = models.Sequential([
    feature_extractor,
    # compress to 10 classes. Embedding of all previous in 10 groups
    layers.Conv2D(10,(1,1), activation="relu", input_shape=(4,4,39)),  
    layers.Conv2D(10*3, (3,3), activation="relu"),  # down sample to 1*1, 3 major features per class
    layers.Conv2D(10*15, (1,1), activation="relu"), # assuming 15 features per class
    layers.Conv2D(10,(1,1), activation="relu"),  # compress to 10 classes again
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])
keras_model.build((28,28,1))
logger.debug(keras_model.summary())

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

# %% [code]

keras_model.compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = ["accuracy", ],
)

keras_model.fit(x_train,y_train, epochs = 1)
keras_model.evaluate(x_val,y_val)

# %% [code]
y_predicted = keras_model.predict(x_client)

y_indexes = np.arange(0, len(y_predicted), 1)
csv_file = open("predictions.csv","w")
csv_file.write("ImageId,Label")
np.savetxt(csv_file, np.concatenate((y_indexes, y_predicted)), delimiter=",")


# %% [markdown]
"""
improve
1. log steps into w&b
2. monitor in tensorboard
3. iteratively improve accuracy & reduce parameter count
3. highest accuracy via feature engineering, with least parameter count. with model explainability
"""
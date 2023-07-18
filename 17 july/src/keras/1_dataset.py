#%%
# exploring keras.datasets & tensorflow_datatasets both
from tensorflow.keras import datasets, preprocessing
import tensorflow_datasets as tfds
# from torchvision import datasets, transforms

import numpy as np
from loguru import logger
# TODO: single element visualize and understand

(x_train, y_train) , (x_test, y_test)  = datasets.mnist.load_data()

train_data = x_train, y_train
test_data = x_test, y_test
logger.debug(f'datatype of x_train_data= {type(x_train)}')
logger.info(f'{x_train.shape}{y_train.shape}')
logger.info(f'expected shape = x = (BCHW), y =  (B,n_classes).  (60000,28,28,1), (60000,10).\n 1 10 dimention vector for 1 example')
assert isinstance(x_train, np.ndarray) # training_data is numpy array not tf.Tensor

#%%
(x_train, y_train) , (x_test, y_test)  = datasets.mnist.load_data()

B,H,W = x_train.shape 
C = 1
x_train = x_train.reshape(B,H,W,C)

B,H,W = x_test.shape
C = 1
x_test = x_test.reshape(B,H,W,C)

B = y_train.shape[0]
n_classes = 10

probs = np.zeros(shape=(B,n_classes))
for index in range(B):
	index_value = y_train[index]
	probs[index][index_value] = 1


#%%
# TUPLE OF ARRAYS
logger.info("1 element, tuple of indexes from 2 arrays")
for index in range(len(x_train)):
	image_x, label_y = x_train[index], y_train[index]
	
	logger.debug(x_train[index].shape)
	logger.debug(y_train[index].shape)
	break
#%%
# TENSORFLOW DATASETS
logger.debug(f'{dir(tfds)}')
# dir(tfds.datasets) vs tfds.dataset_collections() vs vs tfds.list_dataset_collections() vs tfds.list_builders()
logger.debug(f'datasets in TFDS = {len(dir(tfds.list_builders()))}') # 1139 as of now

print("END")

#%%
# B, H, W, C


"""
KEEP IN MIND

1. logging
2. tdd
3. config via yaml
3. docstring
4. modularity & clean coding & codeporn
"""
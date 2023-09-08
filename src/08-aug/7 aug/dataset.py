import os
import numpy as np
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from datetime import datetime

def get_dataset():
    training_data_df = pd.read_csv("../input/digit-recognizer/train.csv")

    y = training_data_df.pop('label')
    X = training_data_df

    y = np.array(y) # y.shape = (42000,). its a scalar hence y
    X = np.array(X)

    X = X/255.0
    X = X.reshape(42000,28,28,1)
    Y = pd.get_dummies(y)
    return X,Y

def get_model():
    keras_model = keras.models.Sequential([
        keras.layers.Input(shape=(28,28,1)), # channels last assumption. 
        # A shape tuple (integers), not including the batch size. Not pythonic to make batch explicit
        keras.layers.Flatten(),
        keras.layers.Dense(40,activation=keras.activations.relu), # relu beautiful. standard naming
        keras.layers.Dense(10,activation=keras.activations.softmax)
    ])
    keras_model.build(input_shape=(1,28,28,1))
    return keras_model

X,Y     = get_dataset()
model   = get_model()

model.compile(
    loss        = "categorical_crossentropy",
    metrics     = ["accuracy"],
    optimizer   = "sgd",
)

parametric_experiment_name = "./logs/mnist"
log_dir = parametric_experiment_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
# REGRESSION BUG
callback_list = [
    keras.callbacks.TensorBoard(log_dir, histogram_freq = 1, write_images=True, update_freq="epoch") # https://github.com/keras-team/keras/issues/16173 update_freq is egnored unless its set to epoch
]

model.fit(X, Y ,epochs=10, callbacks= callback_list, batch_size=32)

"""
upper, lower = np.max(train_X[0]),np.min(train_X[0])

size = upper - lower + 1 # 256 = 8 bits for 1 color. 24 bits for 3 colors
assert size == np.power(2,8)
"""
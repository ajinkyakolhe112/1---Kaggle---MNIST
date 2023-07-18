#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models, layers, activations
# keras is supposed to front end, to abstract away the backend libraries. So in philosophy, its closest to DL terminologies

#%%
from loguru import logger
import numpy as np

# TODO: model parameter & architecture visualization

"""
img: 20*10^6 * 20*10^6 * 3
parameters: filters & kernel size. 
-> RF, Channel Size, Channels

1 Channel -> 1 Neuron -> 1 filter

"""

from tensorflow.keras import datasets, preprocessing

train_data , test_data  = datasets.mnist.load_data()
x_train, y_train = train_data
x_test, y_test = test_data

#%%
features_extractor = tf.keras.models.Sequential([
	layers.Conv2D(filters = 32, kernel_size = (7.0,7.0), activation="relu", input_shape=(28,28,1))
])
#%%
features_extractor = tf.keras.models.Sequential([
	layers.Conv2D(filters = 32, kernel_size = (7,7), activation="relu", input_shape=(28,28,1))
])

#%%
# 4 blocks as layers. Analysis of model & problem
features_extractor = tf.keras.models.Sequential([
	layers.Conv2D(filters = 32, kernel_size = (7,7), activation="relu", input_shape=(28,28,1)),
	layers.Conv2D(filters = 64, kernel_size =(7,7), activation="relu"),
	layers.Conv2D(filters = 256, kernel_size = (7,7), activation="relu"),
	layers.Conv2D(filters = 1024, kernel_size =(7,7), activation="relu"), # 512 feature embedding dimention. all possible once
    # output size = 4*4
])

#%%
decision_maker = tf.keras.models.Sequential([
    layers.Conv2D(filters = 2048, kernel_size = (4,4), input_shape = (4,4,1024), activation="relu"),
    layers.Conv2D(filters = 10*15	, kernel_size =(1,1) , activation="relu"), # 15 features per class. 10 such classes
    layers.Conv2D(filters = 10 	, kernel_size =(1,1), activation="softmax"), # weight = 10 filters of 512 channels each. each creating one channel
])
# functional_model = tf.keras.models.Model(inputs = , outputs = , name = "")
#%%
model = tf.keras.models.Sequential(
    [
        features_extractor,
        decision_maker
    ]
)

logger.info(model(np.random.randn(1,28,28,1)))

#%%
def clean_code(train_data, model):
    feature_extractor = tf.keras.Sequential([
        layers.Conv2D(filters = 32, kernel_size = (7,7), activation="relu", input_shape=(28,28,1)),
	    layers.Conv2D(filters = 64, kernel_size =(7,7), activation="relu"),
	    layers.Conv2D(filters = 256, kernel_size = (7,7), activation="relu"),
	    layers.Conv2D(filters = 1024, kernel_size =(7,7), activation="relu"), 
    ])
    # output: 4*4*1024    
    
    decision_maker = tf.keras.Sequential([
        layers.Conv2D(filters = 512, kernel_size=(4,4), activation="relu"),        
        layers.Conv2D(filters = 10*15, kernel_size=(1,1), activation="relu"),
        layers.Conv2D(filters = 10, kernel_size=(1,1), activation="softmax"),
    ])

    model = tf.keras.models.Sequential([
        features_extractor,
        decision_maker
    ])

    model.fit(x_train,y_train)


#%%
if __name__=="__main__":
    clean_code()

# %%

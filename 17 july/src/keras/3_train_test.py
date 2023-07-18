#%%
import tensorflow as tf
from tensorflow.keras import datasets, preprocessing, models, layers, losses, metrics, optimizers

from loguru import logger
import wandb
import numpy as np


#%%
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

logger.debug(f'{x_train.shape}{y_train.shape}')
B, H, W = x_train.shape
C = 1
n_classes = 10
x_train = x_train.reshape(B,H,W,C)
y_train = y_train.reshape(B,1)
logger.info(f'DATA RESHAPE = {x_train.shape}{y_train.shape}')

y_train_probs = np.zeros(shape=(B,n_classes))
for index in range(B):
	index_value = y_train[index]
	y_train_probs[index][index_value] = 1
#%%

"block kernel = 28/4 = 7"
"delta RF = 6"

feature_extractor = tf.keras.models.Sequential([
    layers.Conv2D(filters = 32, kernel_size = (7,7), input_shape = (28,28,1)), # (H,W,C)
    layers.Activation("relu"),
    layers.Conv2D(filters = 32, kernel_size = (7,7)),
    layers.Activation("relu"),
    layers.Conv2D(filters = 32, kernel_size = (7,7)),
    layers.Activation("relu"),
    layers.Conv2D(filters = 32, kernel_size= (7,7)),
    layers.Activation("relu"),
    ])
test_img = np.random.randn(1,28,28,1)
logger.info(f'OUTPUT SHAPE: {feature_extractor(np.random.randn(1,28,28,1)).shape}')

decision_maker = tf.keras.models.Sequential([
    layers.Conv2D(filters = 10*15, kernel_size = (4,4), input_shape = (4,4, 32)),
    layers.Activation("relu"),
    layers.Conv2D(filters = 10, kernel_size = (1,1)),
    layers.Activation("relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(units = 10),
    layers.Activation("softmax")
])

test_img = np.random.randn(1,4,4,32)
probs = decision_maker(test_img)
logger.info(f'OUTPUT SHAPE: {probs.shape}{probs}')

#%%
model = tf.keras.models.Sequential([
    feature_extractor,
    decision_maker
])
model.compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = ['accuracy']
)


model.fit(x_train,y_train_probs, epochs = 10)
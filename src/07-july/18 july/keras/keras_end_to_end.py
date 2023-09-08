#%%
import tensorflow as tf
from tensorflow.keras import datasets, preprocessing, models, layers, losses, metrics, optimizers

from loguru import logger
import wandb
import numpy as np

#%% [markdown]
"""
- [x] x_train & y_train ready to be used & tested.
- [ ] model & train
- [ ] unit tests, for automatic checking
"""
#%%
train_data, test_data = datasets.mnist.load_data()
x_train, 	y_train = train_data
x_test, 	y_test 	= test_data

logger.debug(f'Current: {x_train.shape}{y_train.shape}')
logger.debug(f'Goal: B*H*W*C, B*10')

B,H,W = x_train.shape
C = 1
n_classes = 10
x_train = x_train.reshape(B,H,W,C)

logger.debug(f'Single Element is single value between 0 to 9')
y_train_probs = np.zeros(shape=(B,n_classes))
element_index = 0
while element_index < B:
	element = y_train[element_index]
	
	assert element <=9 and element >= 0
	y_train_probs[element_index][element] = 1 # Rest are zero.
	element_index = element_index + 1
logger.debug(f'Single Element of y_train_probs {y_train_probs[0]},{y_train[0]}. Only 1 is 1 and rest classes are 0')

B,H,W = x_test.shape
x_test = x_test.reshape(B,H,W,C)
y_test_probs = np.zeros(shape=(B,n_classes))
for index,element in enumerate(y_test):
	assert 0 <= element <= 9
	y_test_probs[index][element] = 1 # Rest are zero
logger.debug(f'Single Element of y_test_probs {y_test_probs[0]},{y_test[0]}. Only 1 is 1 and rest classes are 0')

#%%
kwargs = {
	"activation": "relu",
	"use_bias": False
}
0,0,1,0.6,0
0,0,1,0,0
0,0,1,0,0
0,0,1,0,0
0,0,1,0,0

torch.tensor(5, keep_grads=True)
[1*28*28*1]
feature_extractor = tf.keras.models.Sequential([
	layers.Conv2D(filters = 32,kernel_size = 8, name = "conv1", input_shape = (28,28,1), data_format="channels_last"), # W = (8*8*1) o = (X*W)
	layers.Activation("relu"),
	layers.Conv2D(32,8, name = "conv2"), # o = (X*W1)*W2
	layers.Activation("relu"),
	layers.Conv2D(32,8, name = "conv3"),
	layers.Activation("relu"),
	layers.Conv2D(32,7, name = "conv4"),
	layers.Activation("relu"),
])

decision_maker = tf.keras.models.Sequential([
	layers.Conv2D(10*15,(1,1), name = "compress1", input_shape=(1,1,32), data_format="channels_last"),
	layers.Activation("relu"),
	layers.Conv2D(10,(1,1), name = "compress2"),
	layers.Activation("relu"),
	layers.GlobalAveragePooling2D(),
	layers.Dense(10,name="fc1"),
	layers.Activation("softmax")
])

model = tf.keras.models.Sequential([
	feature_extractor,
	decision_maker
])

test_img = np.random.randn(1,28,28,1)
test_output_probs = model(test_img)
logger.info(f'{test_output_probs.numpy()}. Sum should be 1')

model.summary(expand_nested=True,show_trainable=True)
#%%
model.compile(
	loss="categorical_crossentropy",
	optimizer="sgd",
	metrics =["accuracy"]
)

model.fit(x_train,y_train_probs, epochs = 10)


print("END")
import torch
from torchvision import datasets, transforms
# from tensorflow.keras import datasets, preprocessing

from loguru import logger


train_dataset = datasets.MNIST(download=True,root=".data", train=True)
test_dataset = datasets.MNIST(download=True,root=".data", train=False)

logger.debug(f'{dir(datasets)}{len(dir(datasets))}') # appx 130 datasets

# ARRAY OF TUPLES
logger.info("1 element = tuple of (x,y). Array of elements")
index = 5
x_train,y_train = train_dataset[index]
for batch_num, (PIL_image, y_train) in enumerate(train_dataset):
	image_x, label_y = PIL_image, y_train
	image_tensor = transforms.functional.pil_to_tensor(PIL_image)
	
	logger.debug(image_tensor.shape) # shape is a tensor property. NOT PIL IMAGE PROPERTY
	logger.debug(label_y.shape)

element = train_dataset[0]
logger.debug(element)

print("END")

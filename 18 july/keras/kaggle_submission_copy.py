#%%
import tensorflow as tf
from tensorflow.keras import datasets, preprocessing, models, layers, losses, metrics, optimizers

from loguru import logger
import wandb
import numpy as np
import pandas as pd

#%%
def get_data():
    train_data_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/train.csv", dtype=np.float32)
    test_data_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/test.csv", dtype=np.float32)
    sample_sub_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/sample_submission.csv")
    #%%
    x_train = train_data_df.loc[:,train_data_df.columns != 'label'].values/255.0
    x_test = test_data_df.loc[:,test_data_df.columns != 'label'].values/255.0

    y_train = train_data_df['label'].values

    logger.debug(f'Current: {x_train.shape}{y_train.shape}')
    logger.debug(f'Goal: B*H*W*C, B*10')

    #%%
    B, Px = x_train.shape
    C = 1
    n_classes = 10
    x_train = x_train.reshape(B, 28,28, C)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, C)

    y_train = y_train.reshape(B,1)
    y_train_probs = np.zeros(shape=(B,n_classes))
    for index,value in enumerate(y_train):
        predicted_class_index = int(value.item())
        y_train_probs[index][predicted_class_index] = 1
    
    from sklearn.model_selection import train_test_split
    x_train, y_train_probs, x_val, y_val_probs = train_test_split(x_train, y_train_probs, test_size=0.1, random_state=42)
    
    return x_train, y_train_probs, x_val, y_val_probs, x_test

    
#%%
def get_model():
    feature_extractor = tf.keras.models.Sequential([
        layers.Conv2D(32,4, name = "conv1.1", input_shape = (28,28,1), data_format="channels_last"),
        layers.Activation("relu"),
        layers.Conv2D(32,4, name = "conv1.2", ),
        layers.Activation("relu"),

        layers.Conv2D(32,4, name = "conv2.1"),
        layers.Activation("relu"),
        layers.Conv2D(32,4, name = "conv2.2"),
        layers.Activation("relu"),

        layers.Conv2D(32,4, name = "conv3.1"),
        layers.Activation("relu"),
        layers.Conv2D(32,4, name = "conv3.2"),
        layers.Activation("relu"),

        layers.Conv2D(32,4, name = "conv4.1"),
        layers.Activation("relu"),
        layers.Conv2D(32,4, name = "conv4.2"),
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

    return model
#%%
x_train, y_train_probs, x_val, y_val_probs, x_test = get_data()

model = get_model()

model.compile(
	loss="categorical_crossentropy",
	optimizer="sgd",
	metrics =["accuracy"]
)
x_train, y_train_probs

#%%

model.fit(x_train_final,y_train_probs_final, epochs = 10, batch_size=64)
score = model.evaluate(x_val_final,y_val_probs_final, verbose=1)
val_loss,val_accuracy = score[0], score[1]
logger.info(f'loss= {val_loss}, accuracy = {val_accuracy}')

#%%
predicted_classes = model.predict(x_test)
predicted_classes = np.argmax(predicted_classes, axis=1)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("kaggle-first-submission.csv", index=False, header=True)
print("END")
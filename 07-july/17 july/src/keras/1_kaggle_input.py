#%%
import numpy as np
import pandas as pd


train_data_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/train.csv", dtype=np.float32)
test_data_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/test.csv", dtype=np.float32)


sample_sub_df = pd.read_csv("/Users/ajinkya/Documents/Visual Studio Code/0_PROJECTS/1: Kaggle - MNIST/input/digit-recognizer/sample_submission.csv")
#%%
x_train_data = train_data_df.loc[:,train_data_df.columns != 'label'].values/255.0
x_test_data = test_data_df.loc[:,test_data_df.columns != 'label'].values/255.0

y_train_data = train_data_df['label'].values
#%%
B, Px = x_train_data.shape
C = 1
n_classes = 10
x_train = x_train_data.reshape(B, 28,28, C)
y_train = y_train_data.reshape(B,1)
y_train_probs = np.zeros(shape=(B,n_classes))
for index in range(B):
	index_value = int(y_train[index][0])
	y_train_probs[index][index_value] = 1

logger.debug("done")
# %%

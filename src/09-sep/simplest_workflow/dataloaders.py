import pandas as pd
import numpy as np
import torch
import os
from loguru import logger

train_df = pd.read_csv("./input/digit-recognizer/train.csv")
test_df  = pd.read_csv("./input/digit-recognizer/test.csv")

print(f'{train_df.info()}')

y = train_df['label']
train_df.pop('label')
X = train_df

# Convert X in Pandas -> Numpy -> Torch Tensor
X = torch.tensor(X.to_numpy(), dtype=torch.float32, requires_grad=False)
y = torch.tensor(y.to_numpy(), dtype=torch.int64) # torch.one_hot expects int64, int32 doesn't work
print(f'{X.shape}{y.shape}')

X = X.reshape(42000,28,28,1) 

test_data_X = test_df.to_numpy()
test_data_X = torch.tensor(test_data_X, dtype=torch.float32, requires_grad=False)
test_data_X = test_data_X.reshape(-1,28,28,1)

# Tensor Array to Tensor Dataset format.
train_dataset = torch.utils.data.TensorDataset(X,y) # dataloader always needs datatype = Dataset. Hence we are reformatting
test_dataset  = torch.utils.data.TensorDataset(test_data_X)

print(f'X_train[0] = {train_dataset[0][0].shape}, Y_train[0] = {train_dataset[0][1].shape}')
print(f'X_test[0] = {test_dataset[0][0].shape}')

"Dataset done"

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.7, 0.3])

options_dict = {
    "batch_size": 5,
    "shuffle": True,
}
train_data_loader = torch.utils.data.DataLoader(train_dataset, **options_dict)
val_data_loader   = torch.utils.data.DataLoader(val_dataset, **options_dict)
test_data_loader  = torch.utils.data.DataLoader(test_dataset, **options_dict)



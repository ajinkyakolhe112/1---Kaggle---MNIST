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
X = X.to_numpy()
X = torch.tensor(X, dtype=torch.float32, requires_grad=False)
y = torch.tensor(y)
print(f'{X.shape}{y.shape}')

X = X.reshape(42000,28,28,1) 
y_vec = torch.nn.functional.one_hot(y, 10) # one hot expects, datatype of y to be int 
y_vec = y_vec.to(torch.float32)

test_data_X = test_df.to_numpy()
test_data_X = torch.tensor(test_data_X, dtype=torch.float32, requires_grad=False)
test_data_X = test_data_X.reshape(-1,28,28,1)

# Tensor Array to Tensor Dataset format.
train_dataset = torch.utils.data.TensorDataset(X,y_vec) # dataloader always needs datatype = Dataset. Hence we are reformatting
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


class Baseline_NN(torch.nn.Module):
	def __init__(self, ):
		super().__init__()
		
		self.layer_1        = nn.Linear(in_features= nn_arch[0]  , out_features= nn_arch[1])  # hidden layer 1
		self.layer_2        = nn.Linear(in_features= nn_arch[1]  , out_features= nn_arch[2])  # hidden layer 2
		self.decision_maker = nn.Linear(in_features= nn_arch[2]  , out_features= nn_arch[3])  # output layer
		
		self.relu    = nn.functional.relu
		self.softmax = nn.functional.softmax
		self.log_softmax = nn.functional.log_softmax
		
	def forward(self, x_actual):
		
		x  = x_actual
		x  = x.reshape(-1, 28*28)
		
		z1 = self.layer_1(x)
		a1 = self.relu(z1)
		
		z2 = self.layer_2(a1)
		a2 = self.relu(z2)
		
		z3              = self.decision_maker   (a2)
		softmax_probs   = self.softmax          (z3, dim=1)
		log_softmax     = self.log_softmax      (z3, dim=1)
		
		return log_softmax

model 		= Baseline_NN()
optimizer 	= torch.optim.Adam(model.get_parameters(), lr = 0.01)
loss_func	= torch.nn.functional.cross_entropy

for (batch_no, x_actual, y_actual) in enumerate(train_data_loader):
	y_pred = model(x_actual)
	loss = loss_func(y_pred,y_actual)
	loss.backwards()
	optimizer.step()
	optimizer.zero_grad()
	
	
	
import torch, torch.nn as nn
from dataloaders import *
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

class Baseline_NN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		self.layer_1        = nn.Linear( 28*28 , 4  )  # hidden layer 1
		self.layer_2        = nn.Linear(  4    , 15 )  # hidden layer 2
		self.decision_maker = nn.Linear(  15   , 10 )  # output layer
				
	def forward(self, x_actual):
		
		x  = x_actual
		x  = x.reshape(-1, 28*28)
		
		layer_1_output = self.layer_1(x)
		layer_1_activation = relu(layer_1_output)
		
		layer_2_output = self.layer_2(layer_1_activation)
		layer_2_activation = relu(layer_2_output)
		
		layer_3_output  		= self.decision_maker   (layer_2_activation)
		softmax_probs   		= self.softmax          (layer_3_output, dim=1)
		log_softmax_probs 	= self.log_softmax      (layer_3_output, dim=1)
		
		return log_softmax_probs

model 		= Baseline_NN()
optimizer 	= torch.optim.Adam( model.parameters(), lr = 0.01 )
loss_func	= torch.nn.functional.cross_entropy

for batch_no, (x_actual, y_actual) in enumerate(train_data_loader):
	y_pred = model(x_actual)
	loss = loss_func(y_pred,y_actual)
	loss.backwards()
	optimizer.step()
	optimizer.zero_grad()
	
	
	
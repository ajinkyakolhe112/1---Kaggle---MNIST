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
		softmax_probs   		= softmax          (layer_3_output, dim=1)
		log_softmax_probs 	= log_softmax      (layer_3_output, dim=1)
		
		return softmax_probs, log_softmax_probs

model 		= Baseline_NN()
optimizer 	= torch.optim.Adam( model.parameters(), lr = 0.01 )
loss_func	= torch.nn.functional.cross_entropy
loss_func_log = torch.nn.functional.nll_loss

from torchmetrics.classification import Accuracy

for batch_no, (x_actual, y_actual) in enumerate(train_data_loader):
	
	y_pred_softmax, y_pred_log_softmax = model(x_actual)
	
	torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)
	
	y_actual_labels = torch.argmax(y_actual, dim=1)
	
	loss = loss_func_log(y_pred_log_softmax,y_actual_labels)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	
	accuracy = Accuracy(task ="multiclass", num_classes=10)
	y_actual_labels = torch.argmax(y_actual, dim=1)
	print(accuracy(y_pred_softmax, y_actual_labels))
	print(y_pred_softmax)
	print(f'predicted = {torch.argmax(y_pred_softmax,dim=1)}')
	print(f'actual    = {y_actual_labels}')
	print(accuracy(y_pred_softmax, y_actual_labels))
	
	
	
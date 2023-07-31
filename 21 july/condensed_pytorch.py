import torch
import torch.nn as nn
import pytorch_lightning as thunder

from loguru import logger
import wandb

train_loader, test_loader

class Baseline(nn.Module):
	def __init__(self, neurons_init, kernel_size_init):
		
		self.feature_extractor = nn.ModuleDict({
			nn.Conv2d(neurons_init[0], neurons_init[1], kernel_size_init[0])
			nn.Conv2d(neurons_init[1], neurons_init[2], kernel_size_init[1])
		})
		
		self.model = nn.ModuleDict({
			"extractor": feature_extractor,
			
			nn.Conv2d(512,10,(1,1)),
		})
		
	def forward(self, images_batch):
		pass
		
class Extended(thunder.LightningModule):
	def __init__(self, model):
		self.model = model
		pass
		
	def forward(self):
		pass
	
	def train_step(self):
		pass

	def optimizer_step(self, learning_rate):
		for name,param in self.named_parameters():
			param = param - param.grad * learning_rate
		
	def optimizer(self):
		pass
		
	def backward(self):
		pass
		
thunder.Trainer()

model.fit()
model.evaluate()
model.log()




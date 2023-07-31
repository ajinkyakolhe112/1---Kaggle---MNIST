#%% 
import torch
import torch.nn as nn

from loguru import logger
import wandb

#%%
class BaselineModel(nn.Module):
	def __init__(self):
		prev_ch, current_ch = 0, 0
		
		neurons = (1, 32,64,125, 555)
		filters = neurons
		
		batch_size = 32
		feature_extractor = nn.ModuleDict({
			"conv1": nn.Conv2d(in_channels = neurons[0], out_channels = neurons[1], k = 7,7),
			"conv2": nn.Conv2d(in_channels = neurons[1], out_channels = neurons[2], k = 7,7),
			"conv3": nn.Conv2d(in_channels = neurons[2], out_channels = neurons[3], k = 7,7),
			"conv4": nn.Conv2d(in_channels = neurons[3], out_channels = neurons[4], k = 7,7),
		}
		embedding_features = neurons[4]
		n_classes = 10
		features_per_class = 15
		
		decision_maker = nn.ModuleDict({
			"": nn.Conv2d(embedding_features, n_classes*features_per_class,(1,1)),
			"": nn.Conv2d(n_classes*features_per_class,n_classes*features_per_class, (3,3)),
			"": nn.Conv2d(n_classes*features_per_class,n_classes,(1,1)),
			"": nn.GlobalAveragePooling2d(),
			"": nn.Linear(n_classes,n_classes),
			
		})
		pass

	def forward(self,x_batch):
		pass
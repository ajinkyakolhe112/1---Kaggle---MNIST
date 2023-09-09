import torch, torch.nn as nn
from dataloaders import *
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax

import logging
logging.basicConfig( level = logging.DEBUG, format = '[%(filename)s:%(lineno)d] %(message)s' )
logger = logging.getLogger(__name__)

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
        
        layer_3_output = self.decision_maker   (layer_2_activation)
        
        logits = layer_3_output
        softmax_probs       = softmax    (layer_3_output, dim=1)
        log_softmax_probs 	= log_softmax(layer_3_output, dim=1)
        
        return logits, softmax_probs

model 		= Baseline_NN()
optimizer 	= torch.optim.Adam( model.parameters(), lr = 0.01 )

from torchmetrics.classification import Accuracy

torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

for batch_no, (x_actual, y_scalar_labels) in enumerate(train_data_loader):
    y_pred_logits, y_pred_softmax = model(x_actual)
    loss = torch.nn.functional.cross_entropy(y_pred_softmax, y_scalar_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    accuracy        = Accuracy(task ="multiclass", num_classes=10)
    batch_accuracy  = accuracy(y_pred_softmax, y_scalar_labels)
    
    y_pred_labels = torch.argmax(y_pred_softmax,dim=1)
    logger.debug(f'predicted probs  = {y_pred_softmax}')
    logger.debug(f'predicted labels = {y_pred_labels}')
    logger.debug(f'actual    labels = {y_scalar_labels}')
    logger.debug(f'batch accuracy   = {batch_accuracy}')

	
	
import torch
import torch.nn as nn

from loguru import logger

# Architecture is hardcoded. Number of neurons can be changed
nn_arch = [28*28, 5, 4, 10] 
# Architecture = [ 28*28, 5, 4 , 10 ] 
# 1 input layer = 28*28, 
# 1st hidden layer = 5 neurons, 
# 2nd hidden layer of 4 neurons, 
# 1 output layer with 10 neurons

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

if __name__=="__main__":
    test_single_example()
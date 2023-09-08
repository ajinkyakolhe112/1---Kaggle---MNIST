import torch
import torch.nn as nn

from loguru import logger

# Architecture = [ 28*28, 5, 4 , 10 ] 
# 1 input layer = 28*28, 
# 1st hidden layer = 5 neurons, 
# 2nd hidden layer of 4 neurons, 
# 1 output layer with 10 neurons

class Baseline_NN(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        
        nn_arch = [28*28, 5, 4, 10] # Architecture is hardcoded. Number of neurons can be changed
        
        self.layer_1        = nn.Linear(in_features= nn_arch[0]  , out_features= nn_arch[1])  # hidden layer 1
        self.layer_2        = nn.Linear(in_features= nn_arch[1]  , out_features= nn_arch[2])  # hidden layer 2
        self.decision_maker = nn.Linear(in_features= nn_arch[2]  , out_features= nn_arch[3])  # output layer
        
        self.relu    = nn.functional.relu
        self.softmax = torch.nn.functional.softmax

    
    def forward(self, X_batch):
        X = X_batch
        X = X.reshape(-1, 28*28)
        z1 = self.layer_1(X)
        a1 = self.relu(z1)
        
        z2 = self.layer_2(a1)
        a2 = self.relu(z2)

        z3 = self.decision_maker(a2)
        softmax_output  = self.softmax(z3, dim=1)

        return softmax_output

def test_single_example():
    tmp_img = torch.randn(1,28,28,1)

    model = Baseline_NN()

    predicted_probs = model(tmp_img)
    logger.debug(f'{predicted_probs}')

if __name__=="__main__":
    test_single_example()
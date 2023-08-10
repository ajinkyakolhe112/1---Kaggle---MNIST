import torch
import torch.nn as nn

from loguru import logger


class Baseline_NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1        = nn.Linear(28*28   , 15)
        self.relu           = nn.functional.relu
        self.decision_maker = nn.Linear(15      , 10)
        self.softmax        = nn.Softmax()

    def forward(self, X_batch):
        X = X_batch

        X                = X.reshape(-1,28*28*1)
        tmp_output       = self.layer_1(X)
        activated_output = self.relu(tmp_output)

        tmp_output = self.decision_maker(activated_output)
        softmax_output   = torch.nn.functional.softmax(tmp_output, dim=1)

        return softmax_output

def test_single_example():
    tmp_img = torch.randn(1,28,28,1)

    model = Baseline_NN()

    predicted_probs = model(tmp_img)
    logger.debug(f'{predicted_probs}')

if __name__=="__main__":
    test_single_example()
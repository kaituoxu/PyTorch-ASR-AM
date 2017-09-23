import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMModel(nn.module):

    def __init__(self, ninput, ntarget, nhidden, nlayer):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(ninput, nhidden, nlayer)
        self.fc = nn.Linear(nhidden, ntarget)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        score = self.fc(output)
        return score.view(output.size(0), output.size(1), output.size(2)),hidden

    def init_hidden(self, batch_size):
        weight = next()
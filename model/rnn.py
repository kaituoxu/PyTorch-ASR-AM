import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNCell(nn.Module):
    """
    For studying PyTorch code purpose. 
    My own RNN implementation.
    """

    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev_h):
        next_h = torch.tanh(input.mm(self.weight_ih) +
                            prev_h.mm(self.weight_hh) +
                            self.bias_ih + self.bias_hh)
        return next_h

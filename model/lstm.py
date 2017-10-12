import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    TODO: Doc string.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        """
        Inputs:
        - input_size: D
        - hidden_size: H
        - num_layers: L
        """
        super(LSTM, self).__init__()
        # initilize parameters

    def forward(self, input, hidden):
        """
        Inputs:
        - input: (T, N, D)
        - hidden: (h0, c0), each is (L, N, H)

        Returns a tuple of:
        - output: (T, N, H)
        """
        # T = ..
        # for l in range(L):
        #     for t in range(T):
        #         LSTM_step_forward()


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(input_size,
                                                   4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size,
                                                   4 * hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev):
        """
        Inputs:
        - input: (N, D)
        - prev: (prev_h, prev_c), each is (N, H)
        """
        prev_h, prev_c = prev
        a = input.mm(self.weight_ih) + prev_h.mm(self.weight_hh)  # N x 4H
        if self.bias:
            a += self.bias_ih + self.bias_hh
        ai, af, ag, ao = torch.split(a, self.hidden_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        g = torch.tanh(ag)
        o = torch.sigmoid(ao)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

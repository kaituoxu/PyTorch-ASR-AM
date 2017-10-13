import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    TODO: Doc string.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        Inputs:
        - input_size: D
        - hidden_size: H
        - num_layers: L
        TODO: right now, only support time first. not support batch first
        """
        super(LSTM, self).__init__()
        # remember hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # model parameters
        self.cell = {}
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cell[layer] = LSTMCell(layer_input_size, hidden_size).cuda()  # ugly .cuda()

    def forward(self, input, prev):
        """
        Inputs:
        - input: (T, N, D)
        - prev: (h0, c0), each is (L, N, H)

        Returns a tuple of:
        - output: (T, N, H)
        - prev: (hT, cT), each is (L, N, H)
        """
        T, N, _ = input.size()
        next_hidden = (Variable(torch.zeros(self.num_layers, N, self.hidden_size).cuda()),
                       Variable(torch.zeros(self.num_layers, N, self.hidden_size).cuda()))
        output = {}
        for layer in range(self.num_layers):
            output[layer] = Variable(torch.zeros(T, N, self.hidden_size)).cuda()  # (T, N, H)
            layer_input = input if layer == 0 else output[layer-1]
            layer_prev = prev[0][layer], prev[1][layer]
            for t in range(T):
                layer_prev = self.cell[layer](layer_input[t], layer_prev)
                output[layer][t] = layer_prev[0]
            next_hidden[0][layer], next_hidden[1][layer] = layer_prev
        return output[self.num_layers-1], next_hidden


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
        # print(input.data.type())
        # print(prev[0].data.type())
        # print(self.weight_ih.data.type())
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

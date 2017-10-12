import torch
import torch.nn as nn
from torch.autograd import Variable

import rnn


prnn = nn.RNNCell(10, 20)
mrnn = rnn.RNNCell(10, 20)

mrnn.weight_ih.data = prnn.weight_ih.data.clone().t()
mrnn.weight_hh.data = prnn.weight_hh.data.clone().t()
mrnn.bias_ih.data = prnn.bias_ih.data.clone()
mrnn.bias_hh.data = prnn.bias_hh.data.clone()

input = Variable(torch.randn(6, 3, 10))
hx = Variable(torch.randn(3, 20))
mhx = hx.clone()

output = []
moutput = []
for i in range(6):
    # pytorch
    hx = prnn(input[i], hx)
    output.append(hx)
    # mine
    mhx = mrnn(input[i], mhx)
    moutput.append(mhx)
    # this should approximate 0
    print(mhx-hx)


import torch
import torch.nn as nn
from torch.autograd import Variable

import lstm

L = 2

prnn = nn.LSTM(10, 20, L).cuda()
mrnn = lstm.LSTM(10, 20, L)


mrnn.cell[0].weight_ih.data = prnn.weight_ih_l0.data.clone().t()
mrnn.cell[0].weight_hh.data = prnn.weight_hh_l0.data.clone().t()
mrnn.cell[0].bias_ih.data = prnn.bias_ih_l0.data.clone()
mrnn.cell[0].bias_hh.data = prnn.bias_hh_l0.data.clone()
mrnn.cell[1].weight_ih.data = prnn.weight_ih_l1.data.clone().t()
mrnn.cell[1].weight_hh.data = prnn.weight_hh_l1.data.clone().t()
mrnn.cell[1].bias_ih.data = prnn.bias_ih_l1.data.clone()
mrnn.cell[1].bias_hh.data = prnn.bias_hh_l1.data.clone()

input = Variable(torch.randn(5, 3, 10)).cuda()
h0 = Variable(torch.randn(L, 3, 20)).cuda()
c0 = Variable(torch.randn(L, 3, 20)).cuda()
mh0 = h0.clone()
mc0 = c0.clone()
output, hn = prnn(input, (h0, c0))
moutput, mhn = mrnn(input, (mh0, mc0))

print(output.size())
print(moutput.size())
# this should approximate 0
#print(output-moutput)


import torch
import torch.nn as nn
from torch.autograd import Variable

import lstm


prnn = nn.LSTMCell(10, 20).cuda()
mrnn = lstm.LSTMCell(10, 20).cuda()

mrnn.weight_ih.data = prnn.weight_ih.data.clone().t()
mrnn.weight_hh.data = prnn.weight_hh.data.clone().t()
mrnn.bias_ih.data = prnn.bias_ih.data.clone()
mrnn.bias_hh.data = prnn.bias_hh.data.clone()

input = Variable(torch.randn(6, 3, 10)).cuda()

hx = Variable(torch.randn(3, 20)).cuda()
cx = Variable(torch.randn(3, 20)).cuda()
mhx = hx.clone()
mcx = cx.clone()

output = []
moutput = []
for i in range(6):
    # pytorch
    hx, cx = prnn(input[i], (hx, cx))
    output.append(hx)
    # mine
    mhx, mcx = mrnn(input[i], (mhx, mcx))
    moutput.append(mhx)
    # this should approximate 0
    print(mhx-hx)
#print(output)

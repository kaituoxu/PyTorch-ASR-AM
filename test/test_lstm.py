"""
Check loss is right or not.
When C = 10, the CE loss should approximately equal 2.30 (because the initial 
weight is near zero, and the CE loss should near a fixed number)
"""
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from lstm import LSTMModel


D, C, H, L, N, T = 2, 10, 4, 2, 5, 6

input = Variable(torch.randn(T, N, D).cuda(), requires_grad=False)
output = np.random.randint(0, C, (T, N))
output = Variable(torch.from_numpy(output).long().cuda(), requires_grad=False)

print("="*20 + "input, output" + "="*20)
print(input)
print(output)

model = LSTMModel(D, C, H, L, N).cuda()
loss_func = nn.CrossEntropyLoss()

model.hidden = model.init_hidden()
score = model(input, [1]*N)
print("="*20 + "hidden before reset" + "="*20)
print(model.hidden)
print("="*20 + "hidden after reset" + "="*20)
model.reset_hidden([1]*N)
print(model.hidden)
print("="*20 + "score" + "="*20)
print(score)

print("="*20 + "loss" + "="*20)
loss = loss_func(score.view(-1, C), output.view(-1))
print(loss)

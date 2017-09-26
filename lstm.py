import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):

    def __init__(self, ninput, ntarget, nhidden, nlayer, batch_size):
        super(LSTMModel, self).__init__()
        # model hyper-parameters
        self.nhidden = nhidden
        self.nlayer = nlayer
        self.batch_size = batch_size
        # model component
        self.lstm = nn.LSTM(ninput, nhidden, nlayer)
        self.fc = nn.Linear(nhidden, ntarget)
        # init weights and hiddens
        self.init_weights(init_range=0.1)
        self.hidden = self.init_hidden()

    def forward(self, input, reset_flags):
        self.reset_hidden(reset_flags)
        out, self.hidden = self.lstm(input, self.hidden)
        score = self.fc(out.view(out.size(0) * out.size(1), out.size(2)))
        return score.view(out.size(0), out.size(1), score.size(1))

    def init_hidden(self):
        # MUST call it outside of this file once, before call forward()
        # below is useful, it set hidden on GPU or CPU with model
        weight = next(self.parameters()).data
        h_0 = Variable(weight.new(self.nlayer, self.batch_size, self.nhidden).zero_())
        c_0 = Variable(weight.new(self.nlayer, self.batch_size, self.nhidden).zero_())
        return (h_0, c_0)

    def reset_hidden(self, reset_flags):
        """
        Call this at each minibatch.
        """
        # detach it from history (pytorch mechanics)
        h = Variable(self.hidden[0].data)
        c = Variable(self.hidden[1].data)
        self.hidden = (h, c)
        for b, flag in enumerate(reset_flags):
            if flag == 1:
                self.hidden[0][:, b, :].data.fill_(0)
                self.hidden[1][:, b, :].data.fill_(0)

    def init_weights(self, init_range=0.1):
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    @staticmethod
    def serialize(model, optimizer, epoch):
        # model_is_cuda = next(model.parameters()).is_cuda
        # model = model.module if model_is_cuda else model
        package = {'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict(),
                   'epoch': epoch}
        return package


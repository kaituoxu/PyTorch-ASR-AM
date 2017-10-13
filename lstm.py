import torch
import torch.nn as nn
from torch.autograd import Variable

from model import lstm

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'mylstm': lstm.LSTM
}


class LSTMModel(nn.Module):

    def __init__(self, ninput, ntarget, nhidden, nlayer, rnn_type='lstm'):
        """
        Config the model.

        Inputs:
        - ninput: D
        - ntarget: C
        - nhidden: H
        - nlayer: L
        """
        super(LSTMModel, self).__init__()
        # model hyper-parameters
        self.nhidden = nhidden
        self.nlayer = nlayer
        self.rnn_type = rnn_type
        # model component
        # e.g.: self.rnn = nn.LSTM(ninput, nhidden, nlayer)
        self.rnn = supported_rnns[rnn_type](ninput, nhidden, nlayer)
        self.fc = nn.Linear(nhidden, ntarget)
        # init weights
        self.init_weights(init_range=0.1)

    def forward(self, input, hidden, reset_flags=None, train=False):
        """
        To use nn.DataParallel, all inputs must be torch Variable. N batch will
        be split to multi-GPU. e.g.: using 2 GPU, originate N=16 will be 8 in
        each GPU.

        Inputs:
        - input: Variable, shape = (T, N, D)
        - hidden: Variable, shape = (L, N, H)
        - reset_flags: Variable, shape = (1, N)
        - train: binary, whether is training or not

        Returns a tuple of:
        - score: Variable, shape = (T, N, C)
        - hidden: Variable, shape = (L, N, H)
        """
        if train:
            # Reset hidden states
            hidden = self.reset_hidden(hidden, reset_flags[0])
        out, hidden = self.rnn(input, hidden)
        score = self.fc(out.view(out.size(0) * out.size(1), out.size(2)))
        return score.view(out.size(0), out.size(1), score.size(1)), hidden

    def init_hidden(self, batch_size, cuda=False):
        """ Initialize the LSTM hidden state."""
        # MUST call it once before call model(in, hi, rf)
        weight = next(self.parameters()).data
        h0_tensor = weight.new(self.nlayer, batch_size,
                               self.nhidden).zero_()
        c0_tensor = weight.new(self.nlayer, batch_size,
                               self.nhidden).zero_()
        if cuda:
            h0_tensor = h0_tensor.cuda()
            c0_tensor = c0_tensor.cuda()
        h0 = Variable(h0_tensor)
        c0 = Variable(c0_tensor)
        if self.rnn_type in ['lstm', 'mylstm']:
            return (h0, c0)
        elif self.rnn_type == 'gru':
            return h0
        else:
            print("Not support this type yet.")
            exit(0)

    def reset_hidden(self, hidden, reset_flags):
        """
        Reset the hidden according to the reset_flags.
        Call this at each minibatch in forward().
        Inputs:
        - hidden: Variable, shape = (BPTT steps, N, H)
        - reset_flags: Variable, shape = (N, )
        """
        # detach it from history (pytorch mechanics)
        if self.rnn_type in ['lstm', 'mylstm']:
            h = Variable(hidden[0].data)
            c = Variable(hidden[1].data)
            hidden = (h, c)
            for b, flag in enumerate(reset_flags):
                if flag.data[0] == 1:  # data[0] access the data in Variable
                    hidden[0][:, b, :].data.fill_(0)
                    hidden[1][:, b, :].data.fill_(0)
        elif self.rnn_type == 'gru':
            hidden = Variable(hidden.data)
            for b, flag in enumerate(reset_flags):
                if flag.data[0] == 1:  # data[0] access the data in Variable
                    hidden[:, b, :].data.fill_(0)
        else:
            print("Not support this type yet.")
            exit(0)
        return hidden

    def init_weights(self, init_range=0.1):
        """ Initialize the weights. """
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    @staticmethod
    def serialize(model, optimizer, epoch):
        """ To store more information about model. """
        package = {'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict(),
                   'epoch': epoch}
        return package

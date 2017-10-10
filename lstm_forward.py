import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import kaldi_io
from lstm import LSTMModel

parser = argparse.ArgumentParser(description="ASR AM inference.")
parser.add_argument('--model_path', default='',
                    help='Location of the model to be loaded.')
parser.add_argument('--in_feat', default='',
                    help='kaldi feat respecifier.')
parser.add_argument('--out_feat', default='',
                    help='kaldi feat wespecifier.')
parser.add_argument('--feat_dim', type=int,
                    help='Input feature dimension')
parser.add_argument('--target_dim', type=int,
                    help='Output target dimension')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=3, type=int,
                    help='Number of RNN layers')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='Use cuda to inference')
parser.add_argument('--apply_logsoftmax', dest='apply_logsoftmax',
                    action='store_true',
                    help='Transform NN output by log(softmax())')


def main(args):
    # Load model
    model = LSTMModel(ninput=args.feat_dim,
                      ntarget=args.target_dim,
                      nhidden=args.hidden_size,
                      nlayer=args.hidden_layers)
    if args.cuda:
        model = nn.DataParallel(model, dim=1)  # add .cuda() later
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.eval()  # Turn off Batchnorm & Dropout

    # IO
    feat_rspecifier = args.in_feat
    feat_wspecifier = args.out_feat
    feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(feat_rspecifier)
    feat_writer = kaldi_io.BaseFloatMatrixWriter(feat_wspecifier)

    # Inference each utterance
    for i, ifeat in enumerate(feat_reader):
        utt, mat = ifeat  # mat.shape == (T, D)
        mat = np.expand_dims(mat, axis=1)  # mat.shape -> (T, 1, D), 1 is batch
        print(i, utt, mat.shape, type(mat))

        # Convert to model input
        feat_tensor = torch.FloatTensor(mat)
        if args.cuda:
            pass
        feats = Variable(feat_tensor, requires_grad=False)

        # Reset hidden state
        # hidden = model.module.init_hidden(1, cuda=False)  # 1 is batch_size
        hidden = model.init_hidden(1, cuda=False)  # 1 is batch_size

        # Forward
        scores, _ = model(feats, hidden)  # Tx1xC

        if args.apply_logsoftmax:
            scores = nn.LogSoftmax()(scores)
        out = np.squeeze(scores.data.numpy(), axis=1)  # TxC

        # Write
        feat_writer.write(utt, out)

        if i == 10:
            break


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)

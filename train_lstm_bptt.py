import sys
import numpy as np
import kaldi_io
from data import *
np.set_printoptions(threshold=np.nan, precision=2, linewidth=275)
DEBUG = 1


def train_one_epoch(feats_rspecifier, targets_rspecifier, batch_size, steps):
    """
    Simulate Kaldi nnet-train-multistream.cc
    Args:
        batch_size: number of utterance in parallel
        steps: sequence length of truncated BPTT
    """
    feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(feats_rspecifier)
    target_reader = kaldi_io.RandomAccessInt32VectorReader(targets_rspecifier)

    feat_dim = 40 #feat_reader.next()[1].shape[1]

    feats_utt = [np.array([])] * batch_size # every element is numpy ndarray
    targets_utt = [np.array([])] * batch_size
    new_utt_flags = [0] * batch_size

    x = 0
    while True:
        x += 1
        feat_host, target_host, done = get_one_batch_data(feat_reader,
                                                          target_reader,
                                                          feats_utt, 
                                                          targets_utt,
                                                          new_utt_flags, 
                                                          feat_dim,
                                                          batch_size,
                                                          steps)
        if done: break
        
        # Use feat_host and target_host
        if DEBUG:
            print(feat_host)
            if x == 100: return


def main(args):
    # define Model, Loss, Optimizer here
    # model = LSTMModel(...)
    # loss_func = nn.CrossEntropy(...)
    # optimizer = optim.SGD(...)

    # train model multi-epochs
    for epoch in range(2):
        train_one_epoch(args[1], args[2], int(args[3]), int(args[4]))



if __name__ == "__main__":
    args = sys.argv
    main(args)

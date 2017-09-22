import sys
import numpy as np
import kaldi_io
np.set_printoptions(threshold=np.nan, precision=2, linewidth=275)
DEBUG = 1

def read_one_utt(feat_reader, target_reader):
    while not feat_reader.done():
        # feats composed of (uttid, numpy ndarray)
        feats = feat_reader.next()
        uttid = feats[0]
        if not target_reader.has_key(uttid):
            print(uttid, "missing targets! Skip...")
            continue
        if feats[1].shape[0] != target_reader[uttid].shape[0]:
            print(uttid, "length mismatch between feature and target! Skip...")
            continue
        return uttid, feats[1], target_reader[uttid]
    return None, None, None


def get_one_batch_data(feat_reader, target_reader, feats_utt, targets_utt,
                       new_utt_flags, feat_dim, batch_size, steps):
    #### START prepare mini-batch data ####
    new_utt_flags = [0] * batch_size
    for b in range(batch_size):
        if feats_utt[b].shape[0] == 0:
            uttid, feats, targets = read_one_utt(feat_reader, target_reader)
            if feats is not None:
                feats_utt[b] = feats
                targets_utt[b] = targets
                new_utt_flags[b] = 1
                if DEBUG:
                    print(uttid, feats.shape, targets.shape)

    # end the training after processing all the frames
    frames_to_go = 0
    for b in range(batch_size):
        frames_to_go += feats_utt[b].shape[0]
    if frames_to_go == 0: return None, None, True

    #### START pack the mini-batch data ####
    feat_host = np.zeros((steps, batch_size, feat_dim))
    target_host = np.zeros((steps, batch_size))
    frame_num_utt = [0] * batch_size

    # slice at most 'batch_size' frames
    for b in range(batch_size):
        num_rows = feats_utt[b].shape[0]
        frame_num_utt[b] = min(steps, num_rows)

    # pack the features
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            feat_host[t, b, :] = feats_utt[b][t]

    # pack the targets
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            target_host[t, b] = targets_utt[b][t]
    #### END pack data ####

    # remove the data we just packed
    for b in range(batch_size):
        # feats
        rows = feats_utt[b].shape[0]
        if rows == frame_num_utt[b]:
            feats_utt[b] = np.array([])
        else:
            packed_rows = frame_num_utt[b]
            feats_utt[b] = feats_utt[b][packed_rows:]
            targets_utt[b] = targets_utt[b][packed_rows:]
    #### END prepare mini-batch data ####
    return feat_host, target_host, False


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

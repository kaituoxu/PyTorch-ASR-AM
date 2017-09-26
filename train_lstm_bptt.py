import argparse
import errno
import io
import os
import sys
import time

import numpy as np

import kaldi_io
import torch
import torch.nn as nn
from data import *
from lstm import LSTMModel
from torch.autograd import Variable
from utils import *

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


np.set_printoptions(threshold=np.nan, precision=2, linewidth=275)
DEBUG = 1

parser = argparse.ArgumentParser(description="ASR AM training")
parser.add_argument('--train_feats', default='', help='Train feat kaldi respecifier')
parser.add_argument('--train_targets', default='', help='Train target kaldi respecifier')
parser.add_argument('--val_feats', default='', help='Valid feat kaldi respecifier')
parser.add_argument('--val_targets', default='', help='Valid target kaldi respecifier')
parser.add_argument('--feat_dim', type=int, help='Input feature dimension')
parser.add_argument('--target_dim', type=int, help='Output target dimension')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=3, type=int, help='Number of RNN layers')
parser.add_argument('--bptt_steps', default=20, type=int, help='Sequence length of truncated BPTT')
parser.add_argument('--batch_size', default=10, type=int, help='Batch size for training')
parser.add_argument('--epochs', default=2, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=250, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')


def run_one_epoch(feats_rspecifier, targets_rspecifier, feat_dim, target_dim,
                  cross_valid, model, criterion, batch_size, steps, epoch,
                  max_norm=None, optimizer=None):
    """
    Simulate Kaldi nnet-train-multistream.cc
    Args:
        batch_size: number of utterance in parallel
        steps: sequence length of truncated BPTT
    """
    feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(feats_rspecifier)
    target_reader = kaldi_io.RandomAccessInt32VectorReader(targets_rspecifier)

    feats_utt = [np.array([])] * batch_size  # every element is numpy ndarray
    targets_utt = [np.array([])] * batch_size
    new_utt_flags = [0] * batch_size

    total_correct = 0
    total_loss = 0
    total_frame = 0

    i = 0
    model.hidden = model.init_hidden()  # to use GPU, must do this
    while True:
        i += 1
        feat_host, target_host, new_utt_flags, done = \
            get_one_batch_data(feat_reader, target_reader, feats_utt,
                               targets_utt, new_utt_flags, feat_dim,
                               batch_size, steps)
        if done:
            break

        feat_tensor = torch.FloatTensor(feat_host)
        target_tensor = torch.LongTensor(target_host.astype(np.int64))
        if args.cuda:
            feat_tensor = feat_tensor.cuda()
            target_tensor = target_tensor.cuda()

        feats = Variable(feat_tensor, requires_grad=False)
        targets = Variable(target_tensor, requires_grad=False)

        # Forward
        scores = model(feats, new_utt_flags)  # TxNxC
        # Loss
        scores = scores.view(-1, target_dim)
        loss = criterion(scores, targets.view(-1))
        if not cross_valid:
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
            # Update
            optimizer.step()

        if args.cuda:
            pass
            # print("pass now")
            # torch.cuda.synchronize() # multi-gpu

        _, predict = torch.max(scores, 1)

        total_correct += (predict == targets.view(-1)).sum().data[0]
        total_loss += loss.data[0]
        total_frame += batch_size * steps

        if i % 1000 == 1:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Frame Acc {3:.3f}'.format(epoch + 1, i, total_loss / i,
                                          total_correct / total_frame * 100.0),
                                          flush=True)
        
        # Do I really need this?
        del loss
        del scores
    return total_loss / i, total_correct / total_frame * 100.0


def main(args):
    save_folder = args.save_folder
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # IO
    train_feats_rspecifier = args.train_feats
    train_targets_rspecifier = args.train_targets
    val_feats_rspecifier = args.val_feats
    val_targets_rspecifier = args.val_targets

    feat_dim, target_dim = args.feat_dim, args.target_dim

    # Model
    model = LSTMModel(ninput=feat_dim,
                      ntarget=target_dim,
                      nhidden=args.hidden_size,
                      nlayer=args.hidden_layers,
                      batch_size=args.batch_size
                      )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
    print(model)
    print("Number of parameters: %d" % get_param_size(model))

    if args.cuda:
        model.cuda()
        # Multi-GPU
        # model = torch.nn.DataParallel(model).cuda()

    if args.continue_from:
        print('Loading checkpoint model %s' % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1))
    else:
        start_epoch = 0
    prev_val_loss = float("inf")
    best_val_loss = float("inf")

    # Train model multi-epochs
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        print('Training...')
        model.train()  # Turn on BatchNorm & Dropout
        start = time.time()
        avg_loss, avg_acc =  \
            run_one_epoch(feats_rspecifier=train_feats_rspecifier,
                          targets_rspecifier=train_targets_rspecifier,
                          feat_dim=feat_dim, target_dim=target_dim,
                          model=model, criterion=criterion, cross_valid=False,
                          batch_size=args.batch_size, steps=args.bptt_steps,
                          epoch=epoch, max_norm=args.max_norm,
                          optimizer=optimizer)

        print('-'*80)
        print('Training Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Train Loss {2:.3f} | Train Acc {3:.3f} '.format(
              epoch + 1, time.time() - start, avg_loss, avg_acc))
        print('-'*80)

        if args.checkpoint:
            file_path = '%s/am_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(LSTMModel.serialize(model, optimizer, epoch + 1),
                       file_path)
            print('Saving checkpoint model to %s' % file_path)

        # Cross validation
        print('Cross validation...')
        model.eval()  # Turn off Batchnorm & Dropout
        start = time.time()
        val_loss, val_acc =  \
            run_one_epoch(feats_rspecifier=val_feats_rspecifier,
                          targets_rspecifier=val_targets_rspecifier,
                          feat_dim=feat_dim, target_dim=target_dim,
                          model=model, criterion=criterion, cross_valid=True,
                          batch_size=args.batch_size, steps=args.bptt_steps,
                          epoch=epoch, optimizer=optimizer)
        print('-'*80)
        print('Validation Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Validation Loss {2:.3f} | Validation Acc {3:.3f} '.format(
              epoch + 1, time.time() - start, val_loss, val_acc))
        print('-'*80)

        # Adjust learning rate
        if val_loss >= prev_val_loss:
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = \
                optim_state['param_groups'][0]['lr'] / 2.0
            optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                lr=optim_state['param_groups'][0]['lr']))
        prev_val_loss = val_loss

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_path = os.path.join(save_folder, args.model_path)
            torch.save(LSTMModel.serialize(model, optimizer, epoch + 1),
                       file_path)
            print("Find better validated model, saving to %s" % file_path)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)

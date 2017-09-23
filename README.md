### Aim:
1. Duplicate AISHELL LSTM results using PyTorch.
2. Train GRU.

### Usage
```shell
CUDA_VISIBLE_DEVICES=7 time python train_lstm_bptt.py --train_feats="scp:data/tr_feats.scp" --train_targets="ark,t:data/tr_ali.txt" --feat_dim=40 --target_dim=3019 --cuda
```

### Note:
1. Feat maybe processed by apply-cmvn.
2. Ali must be pdf-id, but not transition-id. Remember to use `ali-to-pdf` to process `ali.JOB.gz`.

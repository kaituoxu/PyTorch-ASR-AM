### Aim:
1. Duplicate AISHELL LSTM results using PyTorch.
2. Train GRU.

### Usage
```bash
CUDA_VISIBLE_DEVICES=7 time nohup python train_lstm_bptt.py --train_feats="scp:data/tr_feats.scp" --train_targets="ark,t:data/tr_ali.txt" --feat_dim=40 --target_dim=3019 --val_feats="scp:data/val_feats.scp" --val_targets="ark,t:data/val_ali.txt" --cuda --checkpoint --epochs=10 > train.log &
```

### Note:
1. Feat maybe processed by apply-cmvn.
```bash
copy-feats scp:data_fbank/train/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:data_fbank/train/utt2spk scp:data_fbank/train/cmvn.scp ark:- "ark,scp:$PWD/data_fbank/train/feats_cmvn.ark,$PWD/data_fbank/train/feats_cmvn.scp"
copy-feats scp:data_fbank/dev/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:data_fbank/dev/utt2spk scp:data_fbank/dev/cmvn.scp ark:- "ark,scp:$PWD/data_fbank/dev/feats_cmvn.ark,$PWD/data_fbank/dev/feats_cmvn.scp"
```
2. Ali must be pdf-id, but not transition-id. Remember to use `ali-to-pdf` to process `ali.JOB.gz`.
```shell
#!/bin/bash

for aligz in `ls ali*gz`; do
    echo $aligz
    echo ${aligz%.gz}
    copy-int-vector "ark:gunzip -c $aligz |" ark,t:${aligz%.gz}.txt
done
cat ali*txt > ali.txt
ali-to-pdf final.mdl ark,t:ali.txt ark,t:ali-pdf.txt
```

### Dependency
#### kaldi-python io wrapper
```bash
$ git clone https://github.com/janchorowski/kaldi-python.git
# follow the instuction in the README of this repo.
```

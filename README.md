### Usage
#### Train
```bash
CUDA_VISIBLE_DEVICES=7 time nohup python train_lstm_bptt.py --train_feats="scp:data/tr_feats_shuf.scp" --train_targets="ark,t:data/tr_ali.txt" --feat_dim=40 --target_dim=3019 --val_feats="scp:data/val_feats.scp" --val_targets="ark,t:data/val_ali.txt" --cuda --checkpoint --epochs=10 > train.log &
```
#### Inference
- [ ] TODO: Fix problem with `nn.DataParallel()`

```bash
python lstm_forward.py --model_path=exp/models-lr.1-clip250-shuf/final.pth.tar --in_feat="scp:data/val_feats_shuf.scp" --out_feat="ark,t:data/val_out.ark" --feat_dim=40 --target_dim=3019 --apply_logsoftmax
```

### Note:
1. Feat maybe be processed by apply-cmvn. And remember to shuf the `.scp` before training with it.

```bash
copy-feats scp:data_fbank/train/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:data_fbank/train/utt2spk scp:data_fbank/train/cmvn.scp ark:- "ark,scp:$PWD/data_fbank/train/feats_cmvn.ark,$PWD/data_fbank/train/feats_cmvn.scp"
copy-feats scp:data_fbank/dev/feats.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:data_fbank/dev/utt2spk scp:data_fbank/dev/cmvn.scp ark:- "ark,scp:$PWD/data_fbank/dev/feats_cmvn.ark,$PWD/data_fbank/dev/feats_cmvn.scp"
shuf feats_cmvn.scp > feats_cmvn_shuf.scp
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

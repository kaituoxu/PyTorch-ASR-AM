This is a set of Python wrappers for Kaldi input-output classes.

It allows you to do e.g.:

```
  In [1]: import kaldi_io
  In [2]: feat_reader = kaldi_io.SequentialBaseFloatMatrixReader('scp:./mfcc/raw_mfcc_test.1.scp')
  In [3]: next(feat_reader)
  Out[3]:
  ('FDHC0_SI1559', Shape: (338, 13)
   [[ 47.97408295 -21.51651001 -24.72166443 ...,  -7.34391451  -5.35192871
       1.24314117]
    [ 46.00983429 -19.34067917 -20.49114227 ...,  -2.23715401  -3.65503502
      -1.64697027]
    [ 43.06345367 -21.29892731 -15.17295933 ...,  -6.0672245  -14.09746265
      -9.02336311]
    ...,
    [ 37.66175842 -27.93688965 -10.73719597 ...,  -4.36497116  -3.1932559
       2.3135519 ]
    [ 38.15282059 -30.81328964 -11.75108433 ...,  -6.77649689  -3.78556442
       2.52763462]
    [ 38.64388275 -29.08744812  -9.59657097 ...,  -1.66973591  -0.54327661
       9.77887821]])
```

# Installation Instructions
The software should build against a recent version of Kaldi:
1. Install boos python. For anaconda linix run `conda install boost`
2. Build and install [Kaldi](https://github.com/kaldi-asr/kaldi), make sure to use `configure --shared`
3. Build kaldi python: `KALDI_ROOT=path_to_kaldi make`
4. Configure PYTHONPATH: ```export PYTHONPATH=`pwd`/kaldi-python```

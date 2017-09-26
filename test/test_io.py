import sys
sys.path.append("..")
import kaldi_io

def main(feat_rspecifier, ali_rspecifier):
    for epoch in range(2):
        feature_reader = kaldi_io.SequentialBaseFloatMatrixReader(feat_rspecifier)
        ali = kaldi_io.RandomAccessInt32VectorReader(ali_rspecifier)

        i = 1
        while not feature_reader.done():
            i += 1
            feat = feature_reader.next()
            print(i, feat[0], feat[1].shape, ali[feat[0]].shape, flush=True)

        # for i, feat in enumerate(feature_reader):
        #     print(i, feat[0], feat[1].shape, ali[feat[0]].shape)
            # Use feat and ali to train NN
            # forward()
            # loss()
            # backward()
            # update()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

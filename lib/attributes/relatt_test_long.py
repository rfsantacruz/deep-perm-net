# Script to test model agains relative attributes dataset
# Author: Rodrigo Santa Cruz
# Date: 4/11/2016
import os

os.environ['GLOG_minloglevel'] = '2'
import sys

pycaffe_path = "/home/rfsc/Projects/deep-perm-net/caffe-perm/python"
if pycaffe_path not in sys.path:
    sys.path.insert(0, pycaffe_path)
import caffe
import numpy as np
import lmdb
import argparse
import cvxpy as cvx

# imagenet mean in bgr
imagenet_mean = np.array([104, 117, 123], np.float32).reshape((3, 1, 1));


class PMAppx:
    def __init__(self, perm_len):
        # define opt problem for inference
        self._rows, self._cols, self._n, = perm_len, perm_len, pow(perm_len, 2)
        self._var = cvx.Bool(self._n)
        self._par = cvx.Parameter(self._n)
        Rnorm = np.fromfunction(lambda i, j: j // self._cols == i, (self._rows, self._n), dtype=int).astype(np.float)
        Cnorm = np.hstack([np.identity(self._cols) for c in range(self._cols)]).astype(np.float)
        self._A = np.vstack((Rnorm, Cnorm))
        self._prob = cvx.Problem(cvx.Minimize(cvx.norm(self._var - self._par)), [self._A * self._var == 1])

    def get_pm(self, dsm):

        self._par.value = dsm
        self._prob.solve()
        if self._prob.status == cvx.OPTIMAL or self._prob.status == cvx.OPTIMAL_INACCURATE:
            pm = np.array(self._var.value, dtype=np.float).ravel()
        else:
            pm = dsm
            print("problem with {}".format(dsm))

        pm = (pm.reshape(self._rows, self._cols).argmax(axis=1)[:, None] == np.arange(self._rows)).flatten() \
            .astype(np.int)
        return pm


def perm_prediction(model, weights, lmdb_path, output_file, seq_len, sub_seq_len, gpu, batch_size, crop_size,
                    seq_stride, feat_rep):
    # load model and inference solver
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(model, weights, caffe.TEST)
    pm_appx = PMAppx(sub_seq_len)

    # load dataset and relevance vector
    reader = lmdb_reader(lmdb_path)

    # Predictions
    dummy_perms = np.tile(np.arange(sub_seq_len), (batch_size, 1))
    batch, gt, pred, rels = 0, [], [], []
    for valids_idx, batch_data, batch_rels in batch_reader(reader, seq_len, batch_size, crop_size):
        # permuting batch
        perms = np.array([np.random.permutation(seq_len) for n in range(batch_data.shape[0])], dtype=np.int)
        batch_data = np.reshape(batch_data, (batch_size, -1, 3, crop_size, crop_size))
        for n, perm in enumerate(perms):
            batch_data[n] = batch_data[n, perm]
        batch_data = np.reshape(batch_data, (batch_size, -1, crop_size, crop_size))

        # forward passes
        pred_perms = np.tile(np.arange(seq_len), (batch_size, 1))
        for n in range(seq_len, 0, -(sub_seq_len - 1)):
            for start in range(0, n - sub_seq_len + 1) if n > sub_seq_len else [0]:
                end = start + sub_seq_len
                sub_seq = batch_data[:, start * 3:end * 3]
                sub_perm = pred_perms[:, start:end]
                out = net.forward(data=sub_seq, imnet_label=dummy_perms)
                sub_seq = np.reshape(sub_seq, (batch_size, -1, 3, crop_size, crop_size))
                for b in range(batch_size):
                    pm = pm_appx.get_pm(out["prob"][b])
                    order = np.dot(pm.reshape(4, 4).T, np.arange(sub_seq_len))
                    sub_seq[b] = sub_seq[b, order]
                    sub_perm[b] = sub_perm[b, order]
                batch_data[:, start * 3:end * 3] = sub_seq.reshape((batch_size, -1, crop_size, crop_size))
                pred_perms[:, start:end] = sub_perm

        # collect results
        pred_pms = np.zeros((batch_size, seq_len, seq_len), dtype=np.float)
        gt_pms = np.zeros((batch_size, seq_len, seq_len), dtype=np.float)
        for n in range(batch_size):
            for r in range(seq_len):
                pred_pms[n, r, pred_perms[n, r]] = 1.0
                gt_pms[n, r, perms[n, r]] = 1.0
            pred_pms[n] = pred_pms[n].T
        pred.append(np.reshape(pred_pms, (batch_size, -1))[:valids_idx])
        gt.append(np.reshape(gt_pms, (batch_size, -1))[:valids_idx])
        rels.append(batch_rels[:valids_idx])

        batch += 1
        if batch % 10 == 0:
            print("{} Batches tested !".format(batch))

    # Save results
    gt = np.vstack(gt)
    print("Ground Truth shape {}".format(gt.shape))
    pred = np.vstack(pred)
    print("Prediction shape {}".format(pred.shape))
    rels = np.vstack(rels)
    print("Relevance shape {}".format(rels.shape))
    assert gt.shape == pred.shape
    assert (gt.shape[0], seq_len) == rels.shape
    np.savez(output_file, gt_pms=gt, gt_rels=rels, pred_dsms=pred)
    # np.save(output_file, np.column_stack((gt, pred)))

    return gt, pred, rels


def centeredCrop(img, crop_size):
    channels, height, width = img.shape;

    top = int(np.floor((height - crop_size) / 2.))
    left = int(np.floor((width - crop_size) / 2.))
    cImg = img[:, top:top + crop_size, left:left + crop_size]

    return cImg


def batch_reader(reader, seq_len, batch_size, crop_size):
    batch = np.zeros((batch_size, seq_len * 3, crop_size, crop_size), np.float32)
    batch_rels = np.zeros((batch_size, seq_len), np.float32)
    idx = 0
    blob_mean = np.tile(imagenet_mean, (seq_len, 1, 1))

    for i, image, rel in reader:
        batch[idx] = centeredCrop(image, crop_size) - blob_mean
        batch_rels[idx] = rel
        idx += 1
        if idx >= batch_size:
            valid = idx
            idx = 0
            yield (valid, batch, batch_rels)

    yield (idx, batch, batch_rels)


def lmdb_reader(fpath):
    lmdb_env = lmdb.open(fpath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    gt_rels = np.load("{}_REL.npz".format(fpath))["gt_rels"]

    for idx, data in enumerate(lmdb_cursor):
        key, value, rels = data[0], data[1], gt_rels[idx]
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, image, rels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Model file")
    parser.add_argument('weights', type=str, help="Weights")
    parser.add_argument('lmdb_path', type=str, help="LMDB test dataset")
    parser.add_argument('output_file', type=str, help="Output file path")
    parser.add_argument('seq_len', type=int, default=20, help="Sequence Length")
    parser.add_argument('sub_seq_len', type=int, default=4, help="Subsequence Length")
    parser.add_argument('--gpu', type=int, default=3, help="GPU number")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--crop_size', type=int, default=227, help="Crop Size")
    parser.add_argument('--seq_stride', type=int, default=1, help="Subsequence Length")
    parser.add_argument('--feat_rep', type=str, default="fc7", help="Feature representation used for pooling")
    args = parser.parse_args()
    print(args)

    perm_prediction(args.model, args.weights, args.lmdb_path, args.output_file, args.seq_len, args.sub_seq_len,
                    args.gpu, args.batch_size, args.crop_size, args.seq_stride, args.feat_rep)

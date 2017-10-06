# Script to test model agains relative attributes dataset
# Author: Rodrigo Santa Cruz
# Date: 4/11/2016


#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Axel Angel, copyright 2015, license GPLv3.
# Adapted by: Rodrigo Santa Cruz
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
from collections import defaultdict

# imagenet mean in bgr
imagenet_mean = np.array([104, 117, 123] ,np.float32).reshape((3,1,1));

def perm_prediction(model, weights, seq_len, lmdb_path, output_file, gpu, batch_size, crop_size, num_perms, sal=False):
	
	# load model
	caffe.set_device(gpu)
	caffe.set_mode_gpu()	
	net = caffe.Net(model, weights, caffe.TEST)

	# load dataset and relevance vector	
	reader = lmdb_reader(lmdb_path)
	
	# Predictions
	batch, gt, pred, rels, diffs = 0, [], [], [], []
	for valids_idx, batch_data, batch_rels in batch_reader(reader, seq_len, batch_size, crop_size):		
		for perm_idx in range(num_perms):
			# perform prediction			
			perms = np.array([np.random.permutation(seq_len) for n in range(batch_data.shape[0])], dtype=np.float)
			out = net.forward(data=batch_data, imnet_label=perms)										
			pred.append(out['prob'][:valids_idx])
			gt.append(out['labels'][:valids_idx])
			rels.append(batch_rels[:valids_idx])
			
			# compute saliency maps
			if sal:
				net.backward(prob=net.blobs['prob'].data, labels=net.blobs['labels'].data)				
				diffs.append(np.array([net.blobs['data_{}'.format(s)].diff[:valids_idx] for s in range(seq_len)]).transpose((1,0,2,3,4)))
	
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
	diffs = np.vstack(diffs) if sal else None
	print("Saliency shape {}".format(diffs.shape if sal else None))
	assert gt.shape == pred.shape
	assert (gt.shape[0], seq_len) == rels.shape
	np.savez(output_file, gt_pms=gt, gt_rels=rels, pred_dsms=pred, sal_map=diffs)
	#np.save(output_file, np.column_stack((gt, pred)))
	
	return gt, pred, rels


def centeredCrop(img, crop_size):

	channels, height, width = img.shape;
	
	top = np.floor((height - crop_size)/2.)
	left = np.floor((width - crop_size)/2.) 
	cImg = img[:, top:top+crop_size, left:left+crop_size]

	return cImg


def batch_reader(reader, seq_len, batch_size, crop_size):
	
	batch = np.zeros((batch_size, seq_len*3, crop_size, crop_size), np.float32)
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
	parser.add_argument('--gpu', type=int, default=3, help="GPU number")	
	parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
	parser.add_argument('--crop_size', type=int, default=227, help="Crop Size")
	parser.add_argument('--seq_len', type=int, default=4, help="Sequence Length")
	parser.add_argument('--num_perms', type=int, default=4, help="Number of Permutations perm sample")
	parser.add_argument('--sal', default=False, action="store_true", help="Compute saliency maps. Add force_backward: true to the model.")
	args = parser.parse_args()
	print(args)


	perm_prediction(args.model, args.weights, args.seq_len, args.lmdb_path, args.output_file, args.gpu, args.batch_size, args.crop_size, args.num_perms, args.sal)
	

	
	


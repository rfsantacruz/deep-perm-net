# Script to create a lmdb data with sampled sequences for supervised learn to rank
# Author: Rodrigo Santa Cruz
# Date: 01/11/2016

import sys
pycaffe_path = "/home/rfsc/Projects/deep-perm-net/caffe-perm/python"
if pycaffe_path not in sys.path:
	sys.path.insert(0, pycaffe_path)
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import numpy as np
import lmdb
import argparse
import scipy.io as scio
import os.path as osp
from itertools import combinations, product, permutations, izip
from PIL import Image
from random import shuffle
import os
import matplotlib.pyplot as plt
import fnmatch
import xml.etree.ElementTree as ET


DATASETS = ["car", "pub_fig", "osr_int", "osr_scene"]
ATTRIBUTES = {DATASETS[3]:["natural", "open", "perspective", "large-objects", "diagonal-plane", "close-depth"]}


def _create_CarDB(db_split, data_dir, output_dir, seq_len=4, num_seq=15e3, strict_order=False):
	
	# Check splits
	if db_split not in ["train", "test"]:
		raise ValueError("Split {} does not exist".format(sb_split))
	
	# Configure Paths
	img_path = osp.join(data_dir, db_split)
	img_antt = osp.join(data_dir, db_split, "antt.xml")

	# Reading annotations
	img_names = fnmatch.filter(os.listdir(img_path), "*.jpg")
	antt_xml_root = ET.parse(img_antt).getroot()
	img_year = np.array([int(antt_xml_root.find("./image/[@filename='{}']".format(img_name)).attrib['year']) for img_name in img_names])
	print("creating {} dataset from {} images".format(db_split, len(img_names)))

	# get image tuples obeying the true order
	int_ord = np.argsort(img_year)
	if db_split == "train" and strict_order:
		img_year[int_ord] = np.arange(1, int_ord.size+1) # To apply stric ordering
	seqs, rels = [], []
	while len(seqs) < num_seq:
		comb = sorted(np.random.choice(np.arange(int_ord.size), seq_len, replace=False))
		comb = tuple([int_ord[i] for i in comb])
		if all([img_year[pv] <= img_year[nx] for pv, nx in combinations(comb, 2)]):
			seqs.append(tuple([img_names[im_idx] for im_idx in comb]))
			rels.append(tuple([img_year[s_idx] for s_idx in comb]))
			if len(seqs) % 1e3 == 0:
				print("{} true ordered tuples sampled...".format(len(seqs)))				
	print("sampled {} true tuples for {} dataset".format(len(seqs), db_split))

	# create lmdb
	print("Creating lmdb dataset...")
	db_path = osp.join(output_dir, "DB{}_ACAR_L{}_N{}_{}_lmdb".format(DATASETS[0], seq_len, num_seq, db_split))
	create_seqdb(db_path, img_path, seqs, rels, (256,256))
	return db_path

def _create_OSRINT(db_split, data_dir, output_dir, seq_len=4, num_seq=15e3, strict_order=False):
	
	# Check splits
	if db_split not in ["train", "test"]:
		raise ValueError("Split {} does not exist".format(sb_split))

	# Set Paths
	ant_path = osp.join(data_dir, "data.mat")
	img_path = osp.join(data_dir, "images")
	ant = scio.loadmat(ant_path)

	# Reading annotation
	int_scores = scio.loadmat(osp.join(data_dir, "interestingness_scores.mat"))
	int_scores = int_scores["interestingness_score"].ravel()				
	im_names = np.array([l[0] for l in np.squeeze(ant["im_names"])])

	is_train = np.load(osp.join(data_dir, "interestingness_partition_75.npy"))
	numsamp = is_train.sum() if db_split == "train" else np.logical_not(is_train).sum()
	print("Split {}: {} samples".format(db_split, is_train.sum()))

	# Ordering the images according to its interestingness

	split_idx = np.equal(is_train, 1) if db_split == "train" else np.equal(is_train, 0)
	ims = im_names[split_idx]
	scores = int_scores[split_idx]		
	int_ord = np.argsort(scores)
	if db_split == "train" and strict_order: # Apply strict ordering
		scores[int_ord] = np.arange(1, int_ord.size+1)

	# get image tuples obeying the true order
	seqs, rels = [], []
	while len(seqs) < num_seq:
		comb = sorted(np.random.choice(np.arange(int_ord.size), seq_len, replace=False))
		comb = tuple([int_ord[i] for i in comb])
		if all([scores[pv] <= scores[nx] for pv, nx in combinations(comb, 2)]):
			seqs.append(tuple([ims[im_idx] for im_idx in comb]))
			rels.append(tuple([scores[s_idx] for s_idx in comb]))
			if len(seqs) % 1e3 == 0:
				print("{} true ordered tuples sampled...".format(len(seqs)))				
	print("sampled {} true tuples for {} dataset".format(len(seqs), db_split))

	# create lmdb
	print("Creating lmdb dataset...")
	db_path = osp.join(output_dir, "DB{}_AINT_L{}_N{}_{}_lmdb".format(DATASETS[2], seq_len, num_seq, db_split))
	create_seqdb(db_path, img_path, seqs, rels)
	return db_path

def _create_PUBOSR(db_type, db_split, data_dir, output_dir, att_names=[], seq_len=4, num_seq=15e3, strict_order=False):
	# Check splits
	if db_split not in ["train", "test"]:
		raise ValueError("Split {} does not exist".format(sb_split))
	
	# Paths	
	ant_path = osp.join(data_dir, "data.mat")
	img_path = osp.join(data_dir, "images")
	ant = scio.loadmat(ant_path)

	att_names = att_names if len(att_names) > 0 else ant['attribute_names']
	att_names = [att_names] if type(att_names) is str else att_names
	for att_name in att_names:
		print("Processing attribute {} ...".format(att_name))

		# Reading Attributes
		_, attIdx = np.where(ant['attribute_names'] == att_name)
		if len(attIdx) == 1:
			attIdx = attIdx[0]
		else:
			print("Error ocurred when search for attribute {}. Please try one of {}.".format(att_name, ant['attribute_names']))
			continue

		# Get true relative ordering
		gt_ords = []
		gt_rels = []		
		att_ord = np.argsort(ant["relative_ordering"][attIdx])
		label_relev =  ant["relative_ordering"][attIdx]
		if db_split == "train" and strict_order:
			label_relev[att_ord] = np.arange(1, att_ord.size+1)
		for comb in permutations(att_ord, seq_len):
			if all([label_relev[pv] <= label_relev[nx] for pv, nx in combinations(comb, 2)]):
				gt_ords.append(comb)
				gt_rels.append(tuple([label_relev[cls] for cls in comb]))			
		print("Computed {} true ordered sequences".format(len(gt_ords)))

		# sample split data				
		in_split = np.squeeze(ant["used_for_training"]) == 1 if db_split == "train" else np.squeeze(ant["used_for_training"]) == 0
		im_names = np.array([l[0] for l in np.squeeze(ant["im_names"])])
		labels = np.squeeze(ant["class_labels"])
		
		db_set, db_rel = [], []
		im_dict = {cls: im_names[np.logical_and(labels == cls, in_split)] for cls in np.unique(labels)}
		for gt_rel, gtOrd, qtdSamp in izip(gt_rels, gt_ords, np.random.multinomial(num_seq, [1.0/len(gt_ords)]*len(gt_ords))):	
			db_set.extend(izip(*[np.random.choice(im_dict[cls+1], size=qtdSamp, replace=True) for cls in gtOrd]))
			db_rel.extend([gt_rel for qtd in range(qtdSamp)])					
		print("Sampled {} sequences for split {}".format(len(db_set), db_split))
					
		# create lmdb dataset
		print("Creating lmdb dataset")
		db_path = osp.join(output_dir, "DB{}_A{}_L{}_N{}_{}_lmdb".format(db_type, att_name, seq_len, num_seq, db_split))
		create_seqdb(db_path, img_path, db_set, db_rel)
	return db_path


def create_db(db_type, db_split, data_dir, output_dir, att_names=[], seq_len=4, num_seq=15e3, strict_order=False):
	db_path = None	

	if db_type == DATASETS[0]:
		# CAR Manufactoring
		print("Creating sequence from {}".format(DATASETS[0]))
		db_path = _create_CarDB(db_split, data_dir, output_dir, seq_len, num_seq, strict_order)

	elif db_type == DATASETS[2]:
		# OSR Interestingness
		print("Creating sequence from {}".format(DATASETS[2]))
		db_path = _create_OSRINT(db_split, data_dir, output_dir, seq_len, num_seq, strict_order)

	elif db_type in [DATASETS[1], DATASETS[3]] :
		# OSR Scenes attributes and Public Figures face attributes
		print("Creating sequence dataset from {}".format(db_type))
		db_path = _create_PUBOSR(db_type, db_split, data_dir, output_dir, att_names, seq_len, num_seq, strict_order)
	else:
		print("Script not implemented to the selected data set: {}".format(db_type))
	
	return db_path


def read_images_from_lmdb(db_name, visualize, seq_len):	
	X, y, idxs = [], [], []
	env = lmdb.open(db_name, readonly=True)
	with env.begin() as txn:
		cursor = txn.cursor()
		for idx, (key, value) in enumerate(cursor):
			datum = caffe_pb2.Datum()
			datum.ParseFromString(value)
			X.append(np.array(datum_to_array(datum)))
			y.append(datum.label)
			idxs.append(idx)
	
	rels = np.load("{}_REL.npz".format(db_name))["gt_rels"]

	if visualize:
		print "Visualizing a few images..."
		for i in range(9):
			img = X[i]
			for seqidx in range(seq_len):			
				im = img[seqidx*3:seqidx*3 + 3, :, :]
				im = im.transpose((1,2,0))
				im = im[:,:,::-1]				
				plt.subplot(1, seq_len, seqidx+1)
				plt.imshow(im)
				plt.title(rels[i, seqidx])
				plt.axis('off')
			plt.show()
	print " ".join(["Reading from", db_name, "done!"])
	return X, y, idxs

def create_seqdb(db_path, root_dir, seqs, rels, resize=None):
	# create a lmdb of sequences stacked at the channel dimensions
	# therefore each blob is (SEQ_LEN*3) X H X W

	# suffle data
	seqs, rels = np.array(seqs), np.array(rels)
	sff = np.random.permutation(seqs.shape[0])
	seqs, rels = seqs[sff], rels[sff]	
	np.savez("{}_REL.npz".format(db_path), gt_rels=rels)

	in_db = lmdb.open(db_path, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn:
		for in_idx, seq in enumerate(seqs):			
			im_seq = None
			for img_fname in seq:
				im = Image.open(osp.join(root_dir, img_fname))
				im = im.convert('RGB') if im.mode == 'L' else im					
				im =  np.array(im.resize(resize, Image.BILINEAR)) if resize else np.array(im)
				im = im[:,:,::-1]	
				im = im.transpose((2,0,1))				
				im_seq = im if im_seq is None else np.append(im_seq, im, axis=0)
			
			im_dat = array_to_datum(im_seq, label=0)
			in_txn.put('{:0>10d}'.format(in_idx).encode('ascii'), im_dat.SerializeToString())
			if in_idx % 1e3 == 0:
				print("{}/{} images saved..".format(in_idx, len(seqs)))

	in_db.close()
	print("Dataset created at {} with {} datums".format(db_path, len(seqs)))
	return db_path


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("db_type", type=str, choices=DATASETS, help="Choose the source dataset")
	parser.add_argument("db_split", type=str, choices=["train", "test"], help="Select Dataset Split")
	parser.add_argument("data_dir", type=str, help="Directory of data source")	
	parser.add_argument("output_dir", type=str, help="Output directory")
	parser.add_argument("att_names", nargs="*", type=str, help="Attribute names")
	parser.add_argument("--seq_len", type=int, default=4, help="Sequence Lenghts")
	parser.add_argument("--num_seq", type=float, default=15e3, help="Number of sequences to use")
	parser.add_argument("--strict_order", default=False, action='store_true', help="Generate sequences using strict ordering, i.e., adopt an order in equality cases.")	
	args = parser.parse_args()
	print(args)

#	read_images_from_lmdb("/home/rfsc/Projects/UnsupLearn-Fast/debug/DBpub_fig_AYoung_L4_N100.0_test_lmdb", True, args.seq_len)
#	args.stop()

	create_db(args.db_type, args.db_split, args.data_dir, args.output_dir, args.att_names, args.seq_len, args.num_seq, args.strict_order)

	



import sys
pycaffe_path = "/home/rfsc/Projects/deep-perm-net/caffe-perm/python"
if pycaffe_path not in sys.path:
	sys.path.insert(0, pycaffe_path)
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import numpy as np
import argparse
import matplotlib.pyplot as plt
import lmdb
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage
#import seaborn as sn


def read_images_from_lmdb(db_name, visualize, seq_len):	
	X = []
	env = lmdb.open(db_name, readonly=True)
	with env.begin() as txn:
		cursor = txn.cursor()
		for idx, (key, value) in enumerate(cursor):
			datum = caffe_pb2.Datum()
			datum.ParseFromString(value)
			img = np.array(datum_to_array(datum))
			
			seq = []
			for seqidx in range(seq_len):			
				im = img[seqidx*3:seqidx*3 + 3, :, :]
				im = im.transpose((1,2,0))
				im = im[:,:,::-1]
				seq.append(im)
			X.append(np.array(seq))						
	X = np.array(X)
	return X

def plot_perm_sample(seq_gt, seq_sh, seq_pred, seq_len):
	fig = plt.figure(1, (4., 4.))
	grid = ImageGrid(fig, 111, nrows_ncols=(3, seq_len), axes_pad=0.2)
	ylabels = ["Ground Truth", "Permuted", "Predicted"]
	for i, seq in enumerate([seq_gt, seq_sh, seq_pred]):
		for s, p in enumerate(range(i*seq_len, i*seq_len + seq_len)):		
			grid[p].imshow(seq[s].astype(np.uint8))
			grid[p].xaxis.set_ticks([]) 
			grid[p].yaxis.set_ticks([]) 
			grid[p].set_ylabel(ylabels[p//seq_len], size=20)

	
	plt.show()


def plot_relatt_sample(seq, seq_len):
	fig = plt.figure(1)
	grid = ImageGrid(fig, 111, nrows_ncols=(1, seq_len), axes_pad=0.2)
	for s in range(seq_len):		
		grid[s].imshow(seq[s].astype(np.uint8))
		grid[s].xaxis.set_ticks([]) 
		grid[s].yaxis.set_ticks([])

	fig.tight_layout()			
	plt.show()

def plot_seq(fig, grid, gidx, seq, seq_len):
	for s in range(seq_len):
		grid[gidx+s].imshow(seq[s].astype(np.uint8))
		grid[gidx+s].xaxis.set_ticks([]) 
		grid[gidx+s].yaxis.set_ticks([])
		grid[gidx+s].axis('off')

def plot_sal_map(fig, grid, gidx, seq_pred, seq_sal, seq_len):
	for s in range(seq_len):
		# preprocess image
		img = seq_pred[s].astype(np.uint8)
		img = skimage.color.rgb2gray(img)		

		# Preprocess saliency		
		sal = seq_sal[s].transpose((1,2,0)).max(axis=-1)
		sal = skimage.transform.resize(sal, img.shape)
		sal = skimage.filters.gaussian_filter(sal, 15)
		sal = np.absolute(sal)
		sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-5)

		# plot saliency map over image
		grid[gidx+s].matshow(sal, alpha=1)
		grid[gidx+s].imshow(img, cmap=plt.cm.gray, alpha=0.5)
		grid[gidx+s].xaxis.set_ticks([]) 
		grid[gidx+s].yaxis.set_ticks([])
		grid[gidx+s].axis('off')



if __name__ == "__main__":
	# arguments 
	parser = argparse.ArgumentParser()
	parser.add_argument('lmdb_path', type=str, help="Lmdb file")
	parser.add_argument('result_file', type=str, help=".npz result file")
	parser.add_argument('seq_len', type=int, help="sequence length")
	parser.add_argument('--gt', default=False, action="store_true", help="Plot ground-truth sequence")
	parser.add_argument('--perm', default=False, action="store_true", help="Plot permuted sequence")
	parser.add_argument('--sal', default=False, action="store_true", help="Plot Salience Map")
	args = parser.parse_args()
	print(args)

	# read infor
	# N * Seq_len * 256 * 256 * 3 (RGB)	
	images = read_images_from_lmdb(args.lmdb_path, False, args.seq_len)
	antt = np.load(args.result_file)	

	for r in range(10):
		i = np.random.randint(images.shape[0])

		# GT
		seq_gt = images[i]
						
		# get perm sequence
		sh = np.dot(antt["gt_pms"][i].reshape(args.seq_len, args.seq_len), np.arange(args.seq_len)).astype(np.int)		
		seq_sh = seq_gt[sh]		

		# get pred sequence
		pred = np.dot(antt["pred_pms"][i].reshape(args.seq_len, args.seq_len).T, sh).astype(np.int)
		seq_pred = seq_gt[pred]

		# get salience map for predicted sequence
		#sal_ord = np.dot(antt["gt_pms"][i].reshape(args.seq_len, args.seq_len).T, sh).astype(np.int)		
		seq_sal = antt["sal_map"][i]
		#seq_sal = seq_sal[pred]		

		# plots
		fig = plt.figure(1)
		g_rows, g_cols, gidx = np.sum([args.gt, args.perm, True, args.sal]), args.seq_len, 0		
		grid = ImageGrid(fig, 111, nrows_ncols=(g_rows, g_cols), axes_pad=0.2)
		if args.gt:
			plot_seq(fig, grid, gidx, seq_gt, g_cols)
			gidx += g_cols
		if args.perm:			
			plot_seq(fig, grid, gidx, seq_sh, g_cols)
			gidx += g_cols
		plot_seq(fig, grid, gidx, seq_pred, g_cols)
		gidx += g_cols
		if args.sal:
			plot_sal_map(fig, grid, gidx, seq_sh, seq_sal, g_cols)
			gidx += g_cols
		fig.tight_layout()
		plt.axis('off')
		plt.show()


# Script to perform relative attribute training and testing
# Author: Rodrigo Santa Cruz
# Date: 04/11/2016

import sys
pycaffe_path = "/home/rfsc/Projects/deep-perm-net/caffe-perm/python"
if pycaffe_path not in sys.path:
	sys.path.insert(0, pycaffe_path)
import caffe
import numpy as np
import argparse
import os.path as osp
import os
from mako.template import Template
from relatt_db import create_db, DATASETS
from mako.template import Template
from mako.lookup import TemplateLookup
from relatt_test import perm_prediction
import glob
import shutil

def transplanting(tgt_model, seq_len, output_path, src_model, src_weights):
	#src_model = "./caffe-perm/models/bvlc_alexnet/deploy.prototxt"
	#src_weights = "./caffe-perm/models/bvlc_alexnet/caffenet.caffemodel"	
		
	imnet = caffe.Net(src_model, src_weights, caffe.TEST)		
	attnet = caffe.Net(tgt_model, caffe.TEST)
	for param_name in imnet.params.keys():
		if param_name not in ["fc6", "fc7", "fc8"]:
			for branch in range(seq_len):
				target_param = "_s".join([param_name, str(branch)])
				print("Transplanting parameter {} to {}".format(param_name, target_param))
				attnet.params[target_param][0].data.flat = imnet.params[param_name][0].data.flat
				attnet.params[target_param][1].data[...] = imnet.params[param_name][1].data
	
	attnet.save(output_path)
	del imnet, attnet

	return output_path

def create_dir(*str_paths):
	path = osp.join(*str_paths)
	os.makedirs(path)	
	return path

def gen_model(model_lookup, model_name, output_dir, **kwargs):
	model_template = model_lookup.get_template(model_name)
	model_str = model_template.render(**kwargs)

	output_file = osp.join(output_dir, ".".join([model_name[:-4], "prototxt"]))	
	with open(output_file, "w") as model_file:
		model_file.write(model_str)

	return output_file

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# arguments 
	parser.add_argument("db_type", type=str, choices=DATASETS, help="Choose the source dataset")	
	parser.add_argument("data_dir", type=str, help="Directory of data source")
	parser.add_argument("model_dir", type=str, help="Directory with solver.prototxt, train_val.prototxt and test.prototxt")
	parser.add_argument("output_dir", type=str, help="Output directory")
	parser.add_argument("att_names", nargs="+", type=str, help="Attribute names")
	
	# options
	parser.add_argument("--gpu", type=int, default=3, help="GPU Number")
	parser.add_argument("--train_val", default=False, action="store_true", help="Train and Validate simutaniously")
	parser.add_argument("--weights", type=str, default=None, help="Path to caffemodel")
	parser.add_argument("--seq_len", nargs="+", type=int, help="Sequence Lenghts")
	parser.add_argument("--num_seq", type=float, nargs=2, default=(10e3, 20e3), help="Number of test and train sequences to use")
	parser.add_argument("--perms_mult", type=int, default=2, help="Number of evaluation permutations per sequence")
	parser.add_argument("--strict_order", default=False, action="store_true", help="Use strict ordering during training")
	parser.add_argument("--fusion_layer", default=False, action="store_true", help="Use fusion layer")

	# Models Options
	parser.add_argument("--src_model", type=str, default="./caffe-perm/models/bvlc_alexnet/deploy.prototxt", help="Solver Validation Iterations")
	parser.add_argument("--src_weights", type=str, default="./caffe-perm/models/bvlc_alexnet/caffenet.caffemodel", help="Solver Validation Iterations")
	parser.add_argument("--test_iters", type=int, default=100, help="Solver Validation Iterations")
	parser.add_argument("--test_int", type=int, default=int(1e3), help="Solver Validation Interval")
	parser.add_argument("--lr", type=float, default=1e-5, help="Solver Base Learning Rate")
	parser.add_argument("--lr_step", type=int, default=int(10e3), help="Solver learning rate step")
	parser.add_argument("--max_iters", type=int, default=int(25e3), help="Solver Max iterations")
	parser.add_argument("--snap_iters", type=int, default=int(5e3), help="Solver Snapshot iterations")
	parser.add_argument("--batch_size", type=int, default=32, help="Train and Test Batch size")
	parser.add_argument("--crop_size", type=int, default=227, help="Train and Test Crop size")
	args = parser.parse_args()
	print(args)

	models = TemplateLookup(directories=[args.model_dir])
	for att_name in args.att_names:
		for seq_len in args.seq_len:
			output_subdir = create_dir(args.output_dir, "A{}".format(att_name), "L{}".format(seq_len))
			print(">>>> Relative Attribute Simulation: Attribute Name {}, Sequence Length {}, Working dir {} <<<<".format(att_name, seq_len, output_subdir))
		
			# Create Datasets
			train_db_path = create_db(args.db_type, "train", args.data_dir, output_subdir, att_name, seq_len, args.num_seq[1], args.strict_order)
			test_db_path = create_db(args.db_type, "test", args.data_dir, output_subdir, att_name, seq_len, args.num_seq[0])
			print("Datasets created at {} and ".format(test_db_path, train_db_path))			
			
			# Create Models and train
			model_train_file = gen_model(models, "trainval.tpl", output_subdir, SEQ_LEN=seq_len, 
			TRAIN_DB_PATH=train_db_path, TEST_DB_PATH=test_db_path, BATCH_SIZE=args.batch_size, CROP_SIZE=args.crop_size, FUSION=args.fusion_layer)
			
			snapshot_prefix = osp.join(create_dir(output_subdir, "models"), "snap") 
			model_solver_file = gen_model(models, "solver.tpl", output_subdir, TRAINVAL_FILE=model_train_file, TRAIN_VAL=args.train_val,
			TEST_ITERS=args.test_iters, TEST_INT=args.test_int, LR=args.lr, LR_STEP=args.lr_step, MAX_ITERS=args.max_iters, 
			SNAPSHOT_ITERS=args.snap_iters, SNAPSHOT_DIR=snapshot_prefix)

			model_test_file = gen_model(models, "test.tpl", output_subdir, SEQ_LEN=seq_len, BATCH_SIZE=args.batch_size, CROP_SIZE=args.crop_size, FUSION=args.fusion_layer)
			print("Training files generated at {}".format([model_train_file, model_solver_file, model_test_file]))
			
			# Load caffe and train
			caffe.set_device(args.gpu)			
			caffe.set_mode_gpu()			
			if args.weights:
				initial_weights = args.weights
			else:
				initial_weights = transplanting(model_train_file, seq_len, "".join([snapshot_prefix, "_imageNet_convs_weights.caffemodel"]), args.src_model, args.src_weights)
			print("Training Model...")						
			caffe_solver = caffe.SGDSolver(model_solver_file)
			caffe_solver.net.copy_from(initial_weights)
			if args.train_val:
				caffe_solver.test_nets[0].copy_from(initial_weights)
			caffe_solver.solve()
			del caffe_solver

			# Test
			print("Testing Model...")					
			test_weights_file = glob.glob("{}*.caffemodel".format(snapshot_prefix))
			test_weights_file.remove(initial_weights)
			test_weights_file = sorted(test_weights_file, key=lambda f: int(filter(str.isdigit, osp.basename(f))))[-1]
			print("Loading Test Weights from {}".format(test_weights_file))
			result_file = osp.join(output_subdir, "result_eval")
			perm_prediction(model_test_file, test_weights_file, seq_len, test_db_path, result_file, args.gpu, args.batch_size, args.crop_size, args.perms_mult)
			print("Result File saved at {}".format(result_file))

			# Clear trash
			shutil.rmtree(train_db_path, ignore_errors=True)
			shutil.rmtree(test_db_path, ignore_errors=True)


			
			






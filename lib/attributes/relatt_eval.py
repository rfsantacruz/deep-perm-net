# Script to evaluate permutation and ranking performance for relative attribute learning
# Author: Rodrigo Santa Cruz
# Date: 05/11/2016

import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import cvxpy as cvx
import itertools
from numpy.core.umath_tests import inner1d
import fnmatch
import os
import re
import pandas as pd


def evaluate(gt_rels, gt_pms, pred_dsms, seq_len, appx=False, prec_pred_pms=None):

	# metrics
	ACC, IHA, CSA, NE, NA, PA, NDCG, KT = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	rows, cols, n, samples = seq_len, seq_len, pow(seq_len, 2), gt_pms.shape[0]
	if prec_pred_pms is None:
		pred_pms = np.zeros(pred_dsms.shape, np.int)
	else:
		pred_pms = prec_pred_pms

	
	# Find the closest permutation matrix from the predicted	
	var = cvx.Bool(n)
	par = cvx.Parameter(n)
	Rnorm =  np.fromfunction(lambda i, j: j//cols == i, (rows, n), dtype=int).astype(np.float)
	Cnorm =  np.hstack([np.identity(cols) for c in range(cols)]).astype(np.float)       
	A = np.vstack((Rnorm, Cnorm))
	prob = cvx.Problem(cvx.Minimize(cvx.norm(var - par)), [A * var == 1])
	appx_count = 1
	if prec_pred_pms is None:
		for idx, dsm in enumerate(pred_dsms):				
			if appx:			
				if appx_count % 100 == 0:
					print("{} samples has benn approximated".format(appx_count))
				appx_count += 1
				
				try:
					par.value = dsm
					prob.solve()
					dsm = np.array(var.value, dtype=np.float).ravel()				
				except:				
					print("Optimization failed with dsm={}".format(dsm))
			pred_pms[idx] = (dsm.reshape(rows, cols).argmax(axis=1)[:, None] == np.arange(seq_len)).flatten().astype(int)
		
			if len(np.unique(pred_pms[idx].reshape(rows, cols).argmax(axis=0))) < seq_len or len(np.unique(pred_pms[idx].reshape(rows, cols).argmax(axis=1))) < seq_len:
				NA += 1.0
	 
	# compute non-accpeted matrix
	NA = NA/samples	

	# Compute Normalization error
	NE = np.mean(np.linalg.norm(1 - np.dot(pred_dsms, A.T), ord=1, axis=1)/A.shape[0])

	# Averaged Cossine Similarity
	dots = inner1d(gt_pms, pred_dsms)
	norms = np.multiply(np.linalg.norm(gt_pms, ord=2, axis=1), np.linalg.norm(pred_dsms, ord=2, axis=1))
	CSA = np.mean(np.divide(dots, norms))	
	
	# Permutation prediction evaluation
	ACC, IHA = perm_eval(gt_pms, pred_pms, seq_len)

	# Ranking Evaluation
	PA, NDCG, KT = rank_eval(gt_rels, gt_pms, pred_pms)		

	return ACC, IHA, CSA, NE, NA, PA, NDCG, KT


def perm_eval(gt_pms, pred_pms, seq_len):
	
	# metrics
	ACC, IHA = 0.0, 0.0
	rows, cols, n, samples = seq_len, seq_len, pow(seq_len,2), gt_pms.shape[0]

	# Compute prediction metrics
	match = np.equal(gt_pms, pred_pms).sum(axis=1)
	ACC = np.mean(np.equal(match, n))
	IHA = np.mean(np.divide(match, n*1.0))	

	return ACC, IHA
	

def rank_eval(gt_rels, gt_pms, pred_pms):

	# metrics
	PA, NDCG, KT = 0.0, 0.0, 0.0
	seq_len, samples = gt_rels.shape[1], gt_pms.shape[0]
	rows, cols = seq_len, seq_len

	for gt_rel, gt_pm, pred_pm in itertools.izip(gt_rels, gt_pms, pred_pms):
		# Compute Predicted Relevance rank		
		pred_rel = np.dot(pred_pm.reshape(rows, cols).T, np.dot(gt_pm.reshape(rows, cols), gt_rel))
		# compute metrics
		PA += pair_acc(gt_rel, pred_rel)
		NDCG += ndcg_score(gt_rel, pred_rel, seq_len)
		KT += kendallTau(gt_rel, pred_rel)

	PA, NDCG, KT = PA/samples, NDCG/samples, KT/samples
	return PA, NDCG, KT


def pair_acc(gt_rel, pred_rel):
	""" Pairwise Accuracy """
	count, total = 0.0, 0.0
	for p, n in itertools.combinations(range(gt_rel.shape[0]), 2):
		total += 1
		if np.sign(gt_rel[n] - gt_rel[p]) == np.sign(pred_rel[n] - pred_rel[p]):
			count += 1
	return count/total


def dcg_score(gt_rel, pred_rel, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k"""
    order = np.argsort(pred_rel)[::-1]
    gt_rel = np.take(gt_rel, order[:k])

    if gains == "exponential":
        gains = np.power(2, gt_rel) - 1
    elif gains == "linear":
        gains = gt_rel
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(gt_rel)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(gt_rel, pred_rel, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters """
    best = dcg_score(gt_rel, gt_rel, k, gains)
    actual = dcg_score(gt_rel, pred_rel, k, gains)
    return actual / best


def kendallTau(gt_rel, pred_rel, ties=False):
	""" Kendall's tau correlation """

	kendall = 0.0
	if ties:
		import scipy.stats as stats
		kendall, _ = stats.kendalltau(gt_rel, pred_rel)
	else:
		l_c, l_d, seq_len = 0.0, 0.0, len(gt_rel)		
		for p, n in itertools.combinations(range(seq_len), 2):
			if np.sign(gt_rel[n] - gt_rel[p]) == np.sign(pred_rel[n] - pred_rel[p]):
				l_c += 1
			else:
				l_d += 1
		kendall = (l_c - l_d) / (0.5 * seq_len * (seq_len - 1))
	return kendall

def isNormalized(dsm, rows, cols):
	is_norm = True
	if len(np.unique(dsm.reshape(rows, cols).argmax(axis=0))) < cols or len(np.unique(dsm.reshape(rows, cols).argmax(axis=1))) < rows:
		is_norm = False
	return is_norm

if __name__ == "__main__":

	# arguments 
	parser = argparse.ArgumentParser()
	parser.add_argument("result_files", nargs="+", type=str, help="Result .npz files with keys: gt_rel, gt_pms, pred_dsms")	
	parser.add_argument("--appx", default=False, action="store_true", help="Solve permutation approximation problem before evaluate. Note: it takes longer!")
	parser.add_argument("--csv", default=None, type=str, help="file path to generate csv")
	args = parser.parse_args()
	print(args)

	# find files
	metrics, metric_names = [], ["Accuracy", "Hamming Distance", "Cossine Similarity", "Normalization Error", "Not DSM", "Pair. Acc", "NDCG", "KT"]
	result_files = fnmatch.filter(args.result_files, "*.npz")
	print("{} result files found".format(result_files))
	for result_file in result_files:

		# read information
		result_dict = dict(np.load(result_file))
		gt_rels = result_dict.get("gt_rels", None) 
		gt_pms = result_dict.get("gt_pms", None)
		pred_dsms = result_dict.get("pred_dsms", None)
		pred_pms = result_dict.get("pred_pms", None)
		seq_len = np.sqrt(gt_pms.shape[1])
		assert round(seq_len) == seq_len
		seq_len = int(seq_len)
		print("\n>>> Evaluating file {} whose sequence lenght is {} <<<".format(result_file, seq_len))
		print("GT relevance shape: {}".format(gt_rels.shape))
		print("GT PMs Mtrices: {}".format(gt_pms.shape))
		print("Pred DSMs Mtrices: {}".format(pred_dsms.shape))
		print("Appx Pred PMs Mtrices: {}".format(pred_pms.shape if pred_pms is not None else 0 ))

		# Evaluate
		ACC, IHA, CSA, NE, NA, PA, NDCG, KT = evaluate(gt_rels, gt_pms, pred_dsms, seq_len, args.appx, pred_pms)
		print("Permutation Metrics:\nAccuracy: {}, Averaged Inverse Hamming Distance: {}, Non well formated matrix: {}, DSM Averaged Normalization L1Error: {}, Averaged Cossine Similarity: {}".format(ACC, IHA, NA, NE, CSA))
		print("Ranking Metrics:\nPaiwise Accuracy: {}, NDCG: {}, Kendall's Tau: {}".format(PA, NDCG, KT))

		metrics.append(np.array([ACC, IHA, CSA, NE, NA, PA, NDCG, KT]))

	print("Results:>>>>>>>>>>>>>>>>>>>>>>>")
	metrics = np.vstack(metrics)	
	df={"file_path": result_files}
	for idx, n in enumerate(metric_names):
		df[n] = metrics[:,idx]
	df = pd.DataFrame(df)
	print(df.to_string())
	if args.csv:
		df.to_csv(args.csv, sep=";")
		
	print("\nAveraging over...")
	means, stds = np.mean(metrics, axis=0), np.std(metrics, axis=0)
	for n, m, s in zip(metric_names, means, stds):
		print("{}: {} ({})".format(n, m, s))		





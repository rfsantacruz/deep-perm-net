# expand to include already approximated

import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import cvxpy as cvx
import itertools
from numpy.core.umath_tests import inner1d
import fnmatch
import os


if __name__ == "__main__":

    # arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+", type=str, help="Result .npz files with keys: gt_rel, gt_pms, pred_dsms")	
    args = parser.parse_args()
    print(args)

    result_files = fnmatch.filter(args.result_files, "*.npz")
    for result_file in args.result_files:
        # read information
        result_dict = dict(np.load(result_file))
        gt_rels = result_dict.get("gt_rels", None) 
        gt_pms = result_dict.get("gt_pms", None)
        pred_dsms = result_dict.get("pred_dsms", None)
	diffs = result_dict.get("sal_map", None)
        seq_len = np.sqrt(gt_pms.shape[1])
        assert round(seq_len) == seq_len
        seq_len = int(seq_len)
        print("\n>>> Evaluating file {} whose sequence lenght is {} <<<".format(result_file, seq_len))
        print("GT relevance shape: {}".format(gt_rels.shape))
        print("GT PMs Mtrices: {}".format(gt_pms.shape))
        print("Pred DSMs Mtrices: {}".format(pred_dsms.shape))

        # Create the permutation matrix approximation problem		
        rows, cols, n, samples = seq_len, seq_len, pow(seq_len, 2), gt_pms.shape[0]
        pred_pms = np.zeros(pred_dsms.shape, np.int)
        var = cvx.Bool(n)
        par = cvx.Parameter(n)
        Rnorm =  np.fromfunction(lambda i, j: j//cols == i, (rows, n), dtype=int).astype(np.float)
        Cnorm =  np.hstack([np.identity(cols) for c in range(cols)]).astype(np.float)       
        A = np.vstack((Rnorm, Cnorm))
        prob = cvx.Problem(cvx.Minimize(cvx.norm(var - par)), [A * var == 1])

        # solve
        for idx, dsm in enumerate(pred_dsms):            
            if idx % 100 == 0:
                print(os.getpid())
                print("{} samples processed".format(idx))

            try:
                par.value = dsm
                prob.solve()
                if prob.status == cvx.OPTIMAL or prob.status == cvx.OPTIMAL_INACCURATE:
                    dsm = np.array(var.value, dtype=np.float).ravel()
                else:
                    print("problem at {}".format(idx))
            except:				
                print("Optimization failed with dsm={}".format(dsm))
            pred_pms[idx] = (dsm.reshape(rows, cols).argmax(axis=1)[:, None] == np.arange(seq_len)).flatten().astype(int)

        # save expandedd version
        save_file = os.path.join(os.path.dirname(result_file), "{}_exp.npz".format(os.path.basename(result_file).split('.')[0]))
        np.savez(save_file ,gt_rels=gt_rels, gt_pms=gt_pms, pred_pms=pred_pms, pred_dsms=pred_dsms, sal_map=diffs)
        print("Expanded file save to {}".format(save_file))




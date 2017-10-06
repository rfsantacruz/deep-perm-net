#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <math.h>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/assign/list_inserter.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <stdio.h>
#include <boost/range/algorithm_ext/push_back.hpp>


#include "caffe/layer.hpp"
#include "caffe/layers/perm_matrix_sequence_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// LayerSetup() handles the number of dimensions;
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels
	const PermMatrixSequenceParameter& param = this->layer_param_.perm_matrix_sequence_param();

	// check bottoms and tops
	CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
	CHECK_EQ(top.size(), param.seq_len() + 1) << "Wrong number of top blobs.";
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "Bottom with different batch_size";

	// check shapes
	CHECK_EQ(bottom[0]->shape(1), 3*param.seq_len()) << "Images must be BGR times the sequence length";
	CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3)) << "Images must have the same height and width";

	// Read subset of permutation to be used
	if (param.perm_subset_size() > 0){
		vector<int> init_shape  = boost::assign::list_of(param.perm_subset_size())(param.seq_len());
		perm_subset.Reshape(init_shape);
		if(boost::filesystem::exists(param.perm_subset_file())){
			std::ifstream input(param.perm_subset_file().c_str());
			for (int b = 0; b < perm_subset.shape(0); b++) {
				for (int p = 0; p < perm_subset.shape(1); p++) {
					int value = 0; input >> value;
					caffe_set(1, static_cast<Dtype>(value), perm_subset.mutable_cpu_data() + perm_subset.offset(b,p));
				}
			}
			input.close();
		}else{
			CHECK_EQ(0, 1) << " if set perm_subset_size you should provide a perm_subset_file for a valid file ";
		}
	}
}

template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Reshape() handles the sizes.
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels

	// reshape patches top blobs
	const PermMatrixSequenceParameter& param = this->layer_param_.perm_matrix_sequence_param();
	vector<int> tops_new_shape = boost::assign::list_of(bottom[0]->shape(0))(static_cast<int>(bottom[0]->shape(1)/param.seq_len()))(bottom[0]->shape(2))(bottom[0]->shape(3));
	for (int i=0; i < top.size() - 1; i++){
		top[i]->Reshape(tops_new_shape);
	}
	// reshape labels top blob to save the permutation matrix performed
	tops_new_shape = boost::assign::list_of(bottom[0]->shape(0))(pow(param.seq_len(), 2));
	top[top.size() - 1]->Reshape(tops_new_shape);
}


template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}


// generate permutations
template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::generate_perms(Blob<Dtype>& batch_perms, Blob<Dtype>& perm_matrix, Blob<Dtype>& target_perms) {

	// find dimensions and zeroing  batch_permutations and labels
	const PermMatrixSequenceParameter& param = this->layer_param_.perm_matrix_sequence_param();
	const int batch_size = batch_perms.shape(0);
	caffe_set(batch_perms.count(), static_cast<Dtype>(0.0), batch_perms.mutable_cpu_data());
	caffe_set(perm_matrix.count(), static_cast<Dtype>(0.0), perm_matrix.mutable_cpu_data());
	int perm_lenght = param.seq_len();

	for(int batch_index = 0; batch_index < batch_size ;  batch_index++){

		// sample or get a permutation
		if (param.do_shuffle()){
			if(param.perm_subset_size() > 0){
				int perm_idx = caffe_rng_rand() % perm_subset.shape(0);
				caffe_copy(perm_subset.count(1), perm_subset.cpu_data() + perm_subset.offset(perm_idx), batch_perms.mutable_cpu_data() + batch_perms.offset(batch_index));
			}else{
				// sample permutation
				vector<Dtype>perm; boost::push_back(perm, boost::irange(0, perm_lenght));
				shuffle(perm.begin(), perm.end());
				caffe_copy(perm.size(), &perm[0], batch_perms.mutable_cpu_data() + batch_perms.offset(batch_index));
			}
		}else{
			caffe_copy(batch_perms.count(1), target_perms.cpu_data() + target_perms.offset(batch_index), batch_perms.mutable_cpu_data() + batch_perms.offset(batch_index));
		}

		//convert permutation to permutation matrix
		for(int i=0; i < perm_lenght; i++){
			int perm_entry = i * perm_lenght + batch_perms.data_at(batch_index,i,0,0);
			caffe_set(1, static_cast<Dtype>(1.0), perm_matrix.mutable_cpu_data() + perm_matrix.offset(batch_index, perm_entry));
		}

	}

}


template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}


INSTANTIATE_CLASS(PermMatrixSequenceLayer);
REGISTER_LAYER_CLASS(PermMatrixSequence);

}  // namespace caffe

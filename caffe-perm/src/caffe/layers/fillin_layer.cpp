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
#include <boost/range/algorithm_ext/push_back.hpp>


#include "caffe/layer.hpp"
#include "caffe/layers/fillin_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
void FillinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// LayerSetup() handles the number of dimensions;
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels
	const FillinParameter& param = this->layer_param_.fillin_param();

	// check bottoms and tops
	CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
	CHECK_EQ(top.size(), pow(param.grid_size(),2)) << "Wrong number of top blobs.";
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "Bottom with different batch_size";

	// check shapes
	CHECK_EQ(bottom[0]->shape(1), 3) << "Images must be BGR";
	CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3)) << "Images must have the same height and width";
	CHECK_GE(bottom[0]->shape(2), param.grid_size()*param.patch_size()) << "Height and width must be larger than (patch_size * grid_size)";
}

template <typename Dtype>
void FillinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Reshape() handles the sizes.
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels

	// reshape patches top blobs
	const FillinParameter& param = this->layer_param_.fillin_param();
	vector<int> tops_new_shape = boost::assign::list_of(bottom[0]->shape(0))(bottom[0]->shape(1))(param.patch_size())(param.patch_size());
	for (int i=0; i < top.size() - 1; i++){
		top[i]->Reshape(tops_new_shape);
	}
	// reshape labels top blob
	tops_new_shape = boost::assign::list_of(bottom[0]->shape(0));
	top[top.size() - 1]->Reshape(tops_new_shape);
}


template <typename Dtype>
void FillinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// This layer does not privide CPU version
	NOT_IMPLEMENTED;
}

// generate permutations
template <typename Dtype>
void FillinLayer<Dtype>::generate_fillin(Blob<int>& batch_fillin, Blob<Dtype>& labels) {

	// find dimensions and zeroing  batch_permutations and labels
	const FillinParameter& param = this->layer_param_.fillin_param();
	const int batch_size = batch_fillin.shape(0);
	const int perm_length = batch_fillin.shape(1);
	caffe_set(batch_fillin.count(), 0, batch_fillin.mutable_cpu_data());
	caffe_set(labels.count(), static_cast<Dtype>(0.0), labels.mutable_cpu_data());

	for(int batch_index = 0; batch_index < batch_size ;  batch_index++){

		//compute fill in task label and patches

		vector<int>perm; boost::push_back(perm, boost::irange(0, perm_length));
		int label = caffe_rng_rand() % perm_length;
		perm.erase(perm.begin() + label); perm.push_back(label);


		//copy to batch data
		caffe_copy(batch_fillin.count(1), &perm[0], batch_fillin.mutable_cpu_data() + batch_fillin.offset(batch_index));
		caffe_set(1, static_cast<Dtype>(label), labels.mutable_cpu_data() + labels.offset(batch_index));
	}
}


template <typename Dtype>
void FillinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// This layer does not privide CPU version
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FillinLayer);
#endif

INSTANTIATE_CLASS(FillinLayer);
REGISTER_LAYER_CLASS(Fillin);

}  // namespace caffe

#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>
#include <boost/assign/list_of.hpp>
#include <string>
#include <boost/lexical_cast.hpp>
#include <thrust/device_vector.h>


#include "caffe/layers/sequence_fusion_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SequenceFusionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const SequenceFusionParameter& param = this->layer_param_.seq_fusion_param();
	caffe_gpu_set(top[0]->count(), Dtype(0.0), top[0]->mutable_gpu_data());
	for (int s = 0; s < _seq_len; ++s)	caffe_gpu_axpy(top[0]->count(), _alpha[s], bottom[s]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void SequenceFusionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const SequenceFusionParameter& param = this->layer_param_.seq_fusion_param();
	for (int s = 0; s < _seq_len; ++s){
		caffe_gpu_set(bottom[s]->count(), Dtype(0.0), bottom[s]->mutable_gpu_diff());
		caffe_gpu_axpy(bottom[s]->count(), _alpha[s], top[0]->gpu_diff(), bottom[s]->mutable_gpu_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SequenceFusionLayer);


}  // namespace caffe

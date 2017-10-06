#include <algorithm>
#include <vector>
#include <math.h>
#include <boost/assign/list_of.hpp>

#include "caffe/layers/sequence_fusion_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SequenceFusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	_seq_len = bottom.size();
	int batch_size = bottom[0]->shape(0);
	_feat_size = bottom[0]->shape(1);
	CHECK_GE(_seq_len, 2) << "Layer must have at least 2 input blobs";
	for (int i = 0; i < _seq_len; ++i) {
		CHECK_EQ(bottom[i]->num_axes(), 2) << "Input blob must be 2D (batch, features)";
		CHECK_EQ(bottom[i]->shape(0), batch_size) << "All the blobs must have the same batch size";
		CHECK_EQ(bottom[i]->shape(1), _feat_size) << "All the blobs must have the same feature size";
		_alpha.push_back(2*(i+1) - _seq_len - 1);
	}
}

template <typename Dtype>
void SequenceFusionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SequenceFusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void SequenceFusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(SequenceFusionLayer);
REGISTER_LAYER_CLASS(SequenceFusion);
}  // namespace caffe

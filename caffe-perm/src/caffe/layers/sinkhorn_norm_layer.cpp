#include <algorithm>
#include <vector>
#include <math.h>
#include <boost/assign/list_of.hpp>

#include "caffe/layers/sinkhorn_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SinkhornNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SinkhornNormParameter& param = this->layer_param_.sinkhorn_norm_param();
	// check if input blob is in right dimenssion
	CHECK_EQ(bottom[0]->num_axes(), 2) << "Input blob should tow dimensions (batch, linearized matrix)";
	CHECK_EQ(bottom[0]->count(1), param.num_rows()*param.num_cols()) << "Linearized matrix should have num_rows * num_cols elements";

	// reshape
	int sum_shape = param.is_row_step() ? bottom[0]->shape(0)*param.num_rows() : bottom[0]->shape(0)*param.num_cols();
	vector<int> shape = boost::assign::list_of(sum_shape);
	sum_norms_.Reshape(shape);
	top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void SinkhornNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SinkhornNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(SinkhornNormLayer);
REGISTER_LAYER_CLASS(SinkhornNorm);
}  // namespace caffe

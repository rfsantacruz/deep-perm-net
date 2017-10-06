#ifndef CAFFE_MULTILABEL_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_MULTILABEL_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class MultilabelMultinomialLogisticLossLayer : public LossLayer<Dtype> {
public:
	explicit MultilabelMultinomialLogisticLossLayer(const LayerParameter& param)
	: LossLayer<Dtype>(param) {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "MultilabelMultinomialLogisticLoss"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  // CAFFE_MULTILABEL_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/multilabel_multinomial_logistic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultilabelMultinomialLogisticLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);

}

template <typename Dtype>
void MultilabelMultinomialLogisticLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	int num = bottom[0]->shape(0);
	int dim = bottom[0]->shape(1);
	Dtype loss = 0;
	for (int i = 0; i < num; ++i) {
		for(int k = 0; k < dim; k++){
			int label = static_cast<int>(bottom[1]->cpu_data()[bottom[1]->offset(i,k)]);
			if( label == 1.0){
				Dtype prob = std::max(bottom[0]->cpu_data()[bottom[0]->offset(i,k)], Dtype(kLOG_THRESHOLD));
				loss -= log(prob);
			}
		}
	}
	top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultilabelMultinomialLogisticLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
	            								   << " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		int num = bottom[0]->shape(0);
		int dim = bottom[0]->shape(1);
		caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
		const Dtype scale = - top[0]->cpu_diff()[0] / num;
		for (int i = 0; i < num; i++) {
			for(int k = 0; k < dim; k++){
				int label = static_cast<int>(bottom[1]->cpu_data()[bottom[1]->offset(i,k)]);
				if(label == 1){
					Dtype prob = std::max(bottom[0]->cpu_data()[bottom[0]->offset(i,k)], Dtype(kLOG_THRESHOLD));
					bottom[0]->mutable_cpu_diff()[bottom[0]->offset(i, k)] = scale / prob;
				}
			}
		}
	}
}

INSTANTIATE_CLASS(MultilabelMultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(MultilabelMultinomialLogisticLoss);

}  // namespace caffe

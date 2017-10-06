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


#include "caffe/layer.hpp"
#include "caffe/layers/jigsaw_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
void JigsawLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// LayerSetup() handles the number of dimensions;
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels
	const JigsawParameter& param = this->layer_param_.jigsaw_param();

	// check bottoms and tops
	CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
	CHECK_EQ(top.size(), pow(param.grid_size(),2) + 1) << "Wrong number of top blobs.";
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "Bottom with different batch_size";

	// check shapes
	CHECK_EQ(bottom[0]->shape(1), 3) << "Images must be BGR";
	CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3)) << "Images must have the same height and width";
	CHECK_GE(bottom[0]->shape(2), param.grid_size()*param.patch_size()) << "Height and width must be larger than (patch_size * grid_size)";

	// Read jigsaw file
	vector<int> init_shape  = boost::assign::list_of(param.max_jigsaw())(pow(param.grid_size(),2));
	jigsaw_set.Reshape(init_shape);
	if(boost::filesystem::exists(param.jigsaw_file())){
		std::ifstream input(param.jigsaw_file().c_str());
		for (int b = 0; b < jigsaw_set.shape(0); b++) {
		    for (int p = 0; p < jigsaw_set.shape(1); p++) {
		    	int value = 0; input >> value;
		    	caffe_set(1, static_cast<int>(value), jigsaw_set.mutable_cpu_data() + jigsaw_set.offset(b,p));
		    }
		}
		input.close();
	}else{
		CHECK_EQ(0, 1) << "Provide a valid jigsaw set file";
	}
}

template <typename Dtype>
void JigsawLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Reshape() handles the sizes.
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels

	// reshape patches top blobs
	const JigsawParameter& param = this->layer_param_.jigsaw_param();
	vector<int> tops_new_shape = boost::assign::list_of(bottom[0]->shape(0))(bottom[0]->shape(1))(param.patch_size())(param.patch_size());
	for (int i=0; i < top.size() - 1; i++){
		top[i]->Reshape(tops_new_shape);
	}
	// reshape labels top blob
	tops_new_shape = boost::assign::list_of(bottom[0]->shape(0));
	top[top.size() - 1]->Reshape(tops_new_shape);
}


template <typename Dtype>
void JigsawLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}


// generate permutations
template <typename Dtype>
void JigsawLayer<Dtype>::generate_jigsaws(Blob<int>& batch_jigsaws, Blob<Dtype>& labels, Blob<Dtype>& target_labels) {

	// find dimensions and zeroing  batch_permutations and labels
	const JigsawParameter& param = this->layer_param_.jigsaw_param();
	const int batch_size = batch_jigsaws.shape(0);
	caffe_set(batch_jigsaws.count(), 0, batch_jigsaws.mutable_cpu_data());
	caffe_set(labels.count(), static_cast<Dtype>(0.0), labels.mutable_cpu_data());


	for(int batch_index = 0; batch_index < batch_size ;  batch_index++){

		int jigsaw_label = -1;
		if (param.do_shuffle()){
			jigsaw_label = caffe_rng_rand() % jigsaw_set.shape(0);
		}else{
			jigsaw_label = target_labels.data_at(batch_index, 0, 0, 0);
		}
		//copy to batch data
		caffe_copy(jigsaw_set.count(1), jigsaw_set.cpu_data() + jigsaw_set.offset(jigsaw_label), batch_jigsaws.mutable_cpu_data() + batch_jigsaws.offset(batch_index));
		caffe_set(1, static_cast<Dtype>(jigsaw_label), labels.mutable_cpu_data() + labels.offset(batch_index));
	}

}


template <typename Dtype>
void JigsawLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(JigsawLayer);
#endif

INSTANTIATE_CLASS(JigsawLayer);
REGISTER_LAYER_CLASS(Jigsaw);

}  // namespace caffe

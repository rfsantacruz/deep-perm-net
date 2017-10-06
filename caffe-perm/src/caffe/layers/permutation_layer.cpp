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


#include "caffe/layer.hpp"
#include "caffe/layers/permutation_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
void PermutationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// LayerSetup() handles the number of dimensions;
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels
	const PermutationParameter& param = this->layer_param_.permutation_param();

	// check bottoms and tops
	CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
	CHECK_EQ(top.size(), pow(param.grid_size(),2) + 1) << "Wrong number of top blobs.";
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "Bottom with different batch_size";

	// check shapes
	CHECK_EQ(bottom[0]->shape(1), 3) << "Images must be BGR";
	CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3)) << "Images must have the same height and width";
	CHECK_GE(bottom[0]->shape(2), param.grid_size()*param.patch_size()) << "Height and width must be larger than (patch_size * grid_size)";

	// generate grid coordinates for CPU implementation
	vector<int> init_shape  = boost::assign::list_of(bottom[0]->shape(0))(pow(param.grid_size(),2))(2);
	coord.Reshape(init_shape);
	this->generate_crops_coord(bottom[0]->shape(2), coord);

}

template <typename Dtype>
void PermutationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// Reshape() handles the sizes.
	// bottom[0] supplies the data
	// bottom[1] supplies the original labels

	// reshape patches top blobs
	const PermutationParameter& param = this->layer_param_.permutation_param();
	vector<int> tops_new_shape = boost::assign::list_of(bottom[0]->shape(0))(bottom[0]->shape(1))(param.patch_size())(param.patch_size());
	for (int i=0; i < top.size() - 1; i++){
		top[i]->Reshape(tops_new_shape);
	}
	// reshape labels top blob
	tops_new_shape = boost::assign::list_of(bottom[0]->shape(0));
	top[top.size() - 1]->Reshape(tops_new_shape);
}


template <typename Dtype>
void PermutationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	//initialize class members
	const PermutationParameter& param = this->layer_param_.permutation_param();
	const int perm_length = int(pow(param.grid_size(),2));
	const int batch_size = bottom[0]->shape(0);
	const int N = static_cast<int>(param.patch_size());
	const int input_size = bottom[0]->shape(2);
	const int patch_pad = floor(input_size / float(param.grid_size())) - param.patch_size();

	// generate permutations and crops coordinates
	vector<int> init_shape = boost::assign::list_of(batch_size)(perm_length);
	Blob<int>permutation(init_shape); this->generate_permutations(permutation, (*top[top.size()-1]));

	// generate random vector and sum to initial coordinates to insert randomness
	Blob<float> crops(coord.shape());
	caffe_rng_uniform(crops.count(), 0.0f, static_cast<float>(patch_pad), crops.mutable_cpu_data());
	caffe_add(crops.count(), crops.cpu_data(), coord.cpu_data(), crops.mutable_cpu_data());

	//crop and permute
	int perm_pos=0, crop_h=0, crop_w=0;
	for(int top_idx = 0; top_idx < top.size()-1; top_idx++){
		for(int batch_idx = 0; batch_idx < top[top_idx]->shape(0); batch_idx++){
			perm_pos = permutation.data_at(batch_idx, top_idx, 0, 0);
			crop_h = floor(crops.data_at(batch_idx,perm_pos,0,0));
			crop_w = floor(crops.data_at(batch_idx,perm_pos,1,0));
			for(int chan_idx = 0; chan_idx < top[top_idx]->shape(1); chan_idx++){
				for(int h_idx = 0; h_idx < top[top_idx]->shape(2);h_idx++){
					// copy contiguously memory
					Dtype* top_dest_data = top[top_idx]->mutable_cpu_data() + top[top_idx]->offset(batch_idx, chan_idx, h_idx, 0);
					const Dtype* bottom_src_data = bottom[0]->cpu_data() + bottom[0]->offset(batch_idx, chan_idx, crop_h + h_idx, crop_w);
					caffe_copy(N, bottom_src_data, top_dest_data);
				}
			}
		}
	}
}


void permute_at_distance(const int distance, const bool identity ,vector<int>& permuted){

	int perm_length = permuted.size();
	if(distance == 1 || (!identity && distance == 0)){
		// for fixed size distance there is no distance equal to one
		return;
	}
	//compute fixed vector and shufflable elements
	int num_fixed = perm_length - distance;
	std::vector<bool> fixed(perm_length, false);
	for(int n = 0; n < num_fixed ; n++)
		fixed[n] = true;
	shuffle(fixed.begin(), fixed.end());
	std::vector<int> shuffle_seq;
	for(int idx=0; idx < perm_length; idx++){
		if(!fixed[idx])
			shuffle_seq.push_back(idx);
	}

	// find new sample
	while(1){
		//shuffle elements
		shuffle(shuffle_seq.begin(), shuffle_seq.end());

		//compute new vector and distance
		int samp_dist = 0;
		for(int seq = 0, shufl_idx = 0; seq < perm_length; seq++){
			if(fixed[seq])
				permuted[seq] = seq;
			else
				permuted[seq] = shuffle_seq[shufl_idx++];

			if (permuted[seq] != seq)
				samp_dist++;
		}

		//check distance
		if(distance == samp_dist)
			break;
	}

}



// generate permutations
template <typename Dtype>
void PermutationLayer<Dtype>::generate_permutations(Blob<int>& batch_permutations, Blob<Dtype>& labels) {

	// find dimensions and zeroing  batch_permutations and labels
	const PermutationParameter& param = this->layer_param_.permutation_param();
	const int batch_size = batch_permutations.shape(0);
	const int perm_length = batch_permutations.shape(1);
	caffe_set(batch_permutations.count(), 0, batch_permutations.mutable_cpu_data());
	caffe_set(labels.count(), static_cast<Dtype>(0.0), labels.mutable_cpu_data());

	int desired_distance = caffe_rng_rand() % (perm_length + 1);
	if (desired_distance == 1 || (!param.identity() && desired_distance == 0))
		desired_distance = 2;

	for(int batch_index = 0; batch_index < batch_size ;  batch_index++){

		//compute permutation and and label
		vector<int>perm(perm_length,-1);
		permute_at_distance(desired_distance, param.identity(), perm);

		//copy to batch data
		caffe_copy(batch_permutations.count(1), &perm[0], batch_permutations.mutable_cpu_data() + batch_permutations.offset(batch_index));
		float label = 0.0f;
		if(param.regression())
				label = desired_distance / float(perm_length);
		else{
			if(param.identity())
				label = desired_distance > 0 ? desired_distance - 1 : desired_distance;
			else
				label = desired_distance - 2;
		}
		caffe_set(1, static_cast<Dtype>(label),labels.mutable_cpu_data() + labels.offset(batch_index));

		//set next distance
		desired_distance = (desired_distance + 1) % (perm_length + 1);
		if (desired_distance == 1 || (!param.identity() && desired_distance == 0))
			desired_distance = 2;
	}

//	//debug: generate fixed transformation(flip) -> 2,1,0,5,4,3,8,7,6
//	vector<int> fixed_transf = boost::assign::list_of(2)(1)(0)(5)(4)(3)(8)(7)(6);
//	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
//		for(int perm_idx = 0; perm_idx < perm_length; perm_idx++){
//			caffe_set(1, fixed_transf[perm_idx], batch_permutations.mutable_cpu_data() + batch_permutations.offset(batch_idx,perm_idx));
//		}
//	}
}

// coordinates generator
template <typename Dtype>
void PermutationLayer<Dtype>::generate_crops_coord(const int blob_side, Blob<float>& batch_crops_cord) {

	// find dimensions
	const PermutationParameter& param = this->layer_param_.permutation_param();
	const int batch_size = batch_crops_cord.shape(0);

	// compute tiles start coordinates
	const int tile_size = floor(blob_side / float(param.grid_size()));
	for(int batch_idx = 0; batch_idx < batch_size; batch_idx++){
		if(batch_idx == 0){
			for(int g_h = 0; g_h < param.grid_size(); g_h++){
				for(int g_w = 0; g_w < param.grid_size(); g_w++ ){
					//set values
					const int perm_pos = (g_h * param.grid_size()) + g_w;
					caffe_set(1, static_cast<float>(g_h * tile_size) ,
							batch_crops_cord.mutable_cpu_data() + batch_crops_cord.offset(batch_idx, perm_pos, 0));
					caffe_set(1, static_cast<float>(g_w * tile_size),
							batch_crops_cord.mutable_cpu_data() + batch_crops_cord.offset(batch_idx, perm_pos, 1));
				}
			}
		}else{
			caffe_copy(batch_crops_cord.count(1), batch_crops_cord.cpu_data() + batch_crops_cord.offset(0),
					batch_crops_cord.mutable_cpu_data() + batch_crops_cord.offset(batch_idx));
		}
	}
}


template <typename Dtype>
void PermutationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//this layer should be used just after data layer and there is no backward pass
	return;
}

#ifdef CPU_ONLY
STUB_GPU(PermutationLayer);
#endif

INSTANTIATE_CLASS(PermutationLayer);
REGISTER_LAYER_CLASS(Permutation);

}  // namespace caffe

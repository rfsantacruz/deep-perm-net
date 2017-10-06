#include <vector>
#include <boost/assign/list_of.hpp>
#include <math.h>
#include <time.h>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>

#include "caffe/layers/permutation_layer.hpp"



namespace caffe {

__global__ void currand_init_kernel(curandState *state, const int seed)
{
	/* Each block gets same seed, a different sequence
	       number, no offset */
    curand_init(seed, threadIdx.x, 0, &state[threadIdx.x]);
}

template <typename Dtype>
__global__ void crop_kernel(const int n, curandState *state, const int top,
		const int grid_size, const int channels, const int patch_size,
		const int src_b_stride, const int src_c_stride, const int src_h_stride,
		const int dst_b_stride, const int dst_c_stride, const int dst_h_stride,
		const int* perms, const Dtype* src, Dtype* dst) {

	// crop coordinate for the block 0->h,1->w
	__shared__ int crop_coord[2];
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < n ){

		//compute coordinates on the target
		int crop_stride = channels*patch_size;
		int dst_b = index/(crop_stride),
			dst_c = (index % crop_stride) / patch_size,
			dst_h = (index % crop_stride) % patch_size;

		if(threadIdx.x == 0){
			//Copy state to local memory for efficiency
			curandState localState = state[blockIdx.x];
			// generate crop coordinates of permuted patch
			int tile_size = floorf(src_h_stride / float(grid_size));
			int patch_pad = tile_size - patch_size;
			int perm_idx = dst_b * powf(grid_size,2) + top;
			int perm_pos = perms[perm_idx];
			crop_coord[0] = ((perm_pos/grid_size) * tile_size) + (int)truncf(curand_uniform(&localState) * (patch_pad + 0.999999));
			crop_coord[1] = ((perm_pos%grid_size) * tile_size) + (int)truncf(curand_uniform(&localState) * (patch_pad + 0.999999));
			/* Copy state back to global memory */
			state[blockIdx.x] = localState;
		}
		__syncthreads();

		// compute source cropped pixels coordinates and make assignment of one line (continuously memory)
		int src_b = dst_b, src_c = dst_c, src_h = dst_h + crop_coord[0], src_w = crop_coord[1];
		for(int w=0; w < patch_size; w++){
			dst[dst_b*dst_b_stride + dst_c*dst_c_stride + dst_h*dst_h_stride + w] = src[src_b*src_b_stride + src_c*src_c_stride + src_h*src_h_stride + src_w + w];
		}
	}
}


template <typename Dtype>
void PermutationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	//initialize class members
	const PermutationParameter& param = this->layer_param_.permutation_param();
	const int perm_length = int(pow(param.grid_size(),2));
	const int grid_size = param.grid_size();
	const int batch_size = bottom[0]->shape(0);
	const int input_size = bottom[0]->shape(2);
	const int patch_size = param.patch_size();
	const int patch_pad = floor(input_size / float(param.grid_size())) - param.patch_size();

	// generate permutations
	vector<int> init_shape = boost::assign::list_of(batch_size)(perm_length);
	Blob<int>permutation(init_shape); this->generate_permutations(permutation, (*top[top.size()-1]));
//	//DEBUG CODE
//	string debug(""); std::ofstream debug_file;
//	debug_file.open ("./distribute_debug.txt", ios::out | ios::app);
//	for(int batch_idx = 0; batch_idx< batch_size; batch_idx++){
//		//debug += "batch = " + boost::lexical_cast<std::string>(batch_idx) + ": ";
//		for(int perm_idx = 0; perm_idx < perm_length; perm_idx++){
//			debug += boost::lexical_cast<std::string>(permutation.data_at(batch_idx, perm_idx, 0, 0));
//		}
//
//		debug += ";" + boost::lexical_cast<std::string>(top[top.size()-1]->data_at(batch_idx,0, 0,0)) + ";\n";
//	}
//	debug_file << debug;
//	debug_file.close();


	//crop and permute
	//compute strides and pointers
	const int src_b_stride = bottom[0]->count(1), src_c_stride = bottom[0]->count(2), src_h_stride = bottom[0]->count(3);
	const int dst_b_stride = top[0]->count(1), dst_c_stride = top[0]->count(2), dst_h_stride = top[0]->count(3);
	//compute threads
	const int NThreads = top[0]->shape(1)*top[0]->shape(2);
	const int NBlocks = top[0]->shape(0);
	const int NValid = NThreads*NBlocks;
	//get pointers to GPU memory
	const int* perm_gpu_data = permutation.gpu_data();
	const Dtype* src_gpu_data = bottom[0]->gpu_data();

	//lunch curand setup kernel, each block has one prng
	if(!isCurandInit){
		isCurandInit = true;
		CUDA_CHECK(cudaMalloc((void **)&devStates, NBlocks * sizeof(curandState)));
		currand_init_kernel<<<1, NBlocks>>>(devStates, time(NULL));
		CUDA_POST_KERNEL_CHECK;
	}

	// lunch crop kernel for each top
	for(int top_idx = 0; top_idx < top.size()-1; top_idx++){
		Dtype* dst_gpu_data = top[top_idx]->mutable_gpu_data();

		crop_kernel<Dtype> <<<NBlocks, NThreads>>>(
				NValid, devStates, top_idx,
				grid_size, top[0]->shape(1), patch_size,
				src_b_stride, src_c_stride, src_h_stride,
				dst_b_stride, dst_c_stride, dst_h_stride,
				perm_gpu_data, src_gpu_data, dst_gpu_data);
		CUDA_POST_KERNEL_CHECK;

	}

}


template <typename Dtype>
void PermutationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}

INSTANTIATE_LAYER_GPU_FUNCS(PermutationLayer);

}  // namespace caffe

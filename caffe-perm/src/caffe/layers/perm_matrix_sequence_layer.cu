#include <vector>
#include <boost/assign/list_of.hpp>
#include <math.h>
#include <time.h>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>

#include "caffe/layers/perm_matrix_sequence_layer.hpp"



namespace caffe {


template <typename Dtype>
__global__ void perm_matrix_sequence_kernel(const int n, const int top, const int img_size, const int seq_length,
		const int src_b_stride, const int src_c_stride, const int src_h_stride,
		const int dst_b_stride, const int dst_c_stride, const int dst_h_stride,
		const Dtype* perms, const Dtype* src, Dtype* dst) {

	// crop coordinate for the block 0->h,1->w
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < n ){

		//compute coordinates on the target
		int dst_b = index/(3*img_size),
			dst_c = (index % (3*img_size)) / img_size,
			dst_h = (index % (3*img_size)) % img_size;

		int ch_shift = perms[dst_b * seq_length + top] * 3;

		// compute source cropped pixels coordinates and make assignment of one line (continuously memory)
		int src_b = dst_b, src_c = dst_c + ch_shift, src_h = dst_h, src_w = 0;
		for(int w=0; w < img_size; w++){
			dst[dst_b*dst_b_stride + dst_c*dst_c_stride + dst_h*dst_h_stride + w] = src[src_b*src_b_stride + src_c*src_c_stride + src_h*src_h_stride + src_w + w];
		}
	}
}


template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	//initialize class members
	const PermMatrixSequenceParameter& param = this->layer_param_.perm_matrix_sequence_param();
	const int perm_length = int(param.seq_len());
	const int batch_size = bottom[0]->shape(0);
	const int input_size = bottom[0]->shape(2);

	// generate permutations
	vector<int> init_shape = boost::assign::list_of(batch_size)(perm_length);
	Blob<Dtype>perms(init_shape); this->generate_perms(perms, (*top[top.size()-1]), (*bottom[1]));
//	//DEBUG CODE
//	string debug(""); std::ofstream debug_file;
//	debug_file.open ("./distribute_debug.txt", ios::out | ios::app);
//	for(int batch_idx = 0; batch_idx< batch_size; batch_idx++){
//		//debug += "batch = " + boost::lexical_cast<std::string>(batch_idx) + ": ";
//		for(int perm_idx = 0; perm_idx < perm_length; perm_idx++){
//			debug += boost::lexical_cast<std::string>(perms.data_at(batch_idx, perm_idx, 0, 0));
//		}
//		debug += " ; ";
//		for(int perm_matrix_entry = 0; perm_matrix_entry < perm_length*perm_length; perm_matrix_entry++){
//			debug += boost::lexical_cast<std::string>(top[top.size()-1]->data_at(batch_idx,perm_matrix_entry, 0,0));
//		}
//		debug += "\n";
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
	const Dtype* perm_gpu_data = perms.gpu_data();
	const Dtype* src_gpu_data = bottom[0]->gpu_data();

	// lunch shuffle kernel for each top
	for(int top_idx = 0; top_idx < top.size()-1; top_idx++){
		Dtype* dst_gpu_data = top[top_idx]->mutable_gpu_data();
		perm_matrix_sequence_kernel<Dtype> <<<NBlocks, NThreads>>>(
				NValid, top_idx, top[0]->shape(3), perm_length,
				src_b_stride, src_c_stride, src_h_stride,
				dst_b_stride, dst_c_stride, dst_h_stride,
				perm_gpu_data, src_gpu_data, dst_gpu_data);
		CUDA_POST_KERNEL_CHECK;
	}

}


template <typename Dtype>
void PermMatrixSequenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	return;
}

INSTANTIATE_LAYER_GPU_FUNCS(PermMatrixSequenceLayer);

}  // namespace caffe

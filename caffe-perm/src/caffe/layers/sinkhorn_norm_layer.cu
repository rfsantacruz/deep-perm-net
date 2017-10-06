#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>
#include <boost/assign/list_of.hpp>
#include <string>
#include <boost/lexical_cast.hpp>
#include <thrust/device_vector.h>


#include "caffe/layers/sinkhorn_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RowNorm_Grad_kernel(const int num, const int num_rows, const int num_cols, const Dtype lambda, const Dtype* input, const Dtype* sums, const Dtype* top_diff, Dtype* bottom_diff) {

	CUDA_KERNEL_LOOP(idx,num){
		int batch_stride = num_rows * num_cols;
		int row_stride = num_cols;

		int b = idx / batch_stride;
		int p = (idx % batch_stride) /  row_stride;
		int q = ((idx % batch_stride) %  row_stride);

		Dtype sum = sums[(b*num_rows) + p];
		Dtype sq_sum = pow(sum,2);
		Dtype acc = 0.0;
		for(int j=0; j < num_cols; j++){
			Dtype ind = (j == q) ? Dtype(1.0) : Dtype(0.0);
			acc +=  (((ind/sum) - ((input[(b*batch_stride) + (p*row_stride) + j] + lambda)/sq_sum)) * top_diff[(b*batch_stride) + (p*row_stride) + j]);
		}
		bottom_diff[idx] = acc;

	}
}

template <typename Dtype>
__global__ void ColNorm_Grad_kernel(const int num, const int num_rows, const int num_cols, const Dtype lambda, const Dtype* input, const Dtype* sums, const Dtype* top_diff, Dtype* bottom_diff) {

	CUDA_KERNEL_LOOP(idx,num){
		int batch_stride = num_rows * num_cols;
		int row_stride = num_cols;

		int b = idx / batch_stride;
		int p = (idx % batch_stride) /  row_stride;
		int q = ((idx % batch_stride) %  row_stride);

		Dtype sum = sums[b*num_cols + q];
		Dtype sq_sum = pow(sum, 2);
		Dtype acc = 0.0;
		for(int i =0; i < num_rows; i++){
			Dtype ind = (i == p) ? Dtype(1.0) : Dtype(0.0);
			acc += (((ind/sum) - ((input[(b*batch_stride) + (i*row_stride) + q] + lambda)/sq_sum))* top_diff[(b*batch_stride) + (i*row_stride) + q]);
		}
		bottom_diff[idx] = acc;

	}
}


template <typename Dtype>
__global__ void Trans_Tensor_kernel(const int num, const int num_rows, const int num_cols, const Dtype* input, Dtype* output) {

	CUDA_KERNEL_LOOP(i,num){
		int batch_stride = num_rows * num_cols;
		int row_stride = num_cols;

		int b = i / batch_stride;
		int r = (i % batch_stride) /  row_stride;
		int c = (i % batch_stride) %  row_stride;
		output[i] = input[(b * batch_stride) + (c * row_stride) + r];
	}
}

template <typename Dtype>
void SinkhornNormLayer<Dtype>::ColNorm_GPU(const int num_rows, const int num_cols, const Dtype lambda, Blob<Dtype>& input,  Blob<Dtype>& sums, Blob<Dtype>& output){

	// dimensions and aux vectors
	int num_els =  input.count();
	int num_batch = input.shape(0);

	thrust::device_vector<Dtype> buffer(num_els, 0);
	thrust::device_vector<Dtype> ones(num_rows, 1);
	thrust::device_vector<Dtype> col_sums(num_batch*num_cols, 0);

	//compute col sums - see the input blob 256 x 16 as a list of lines (1024x4) - transpose to get a list of columns 1024*4
	Trans_Tensor_kernel<<<CAFFE_GET_BLOCKS(num_els),CAFFE_CUDA_NUM_THREADS>>>(
				num_els, num_rows, num_cols, input.gpu_data(), thrust::raw_pointer_cast(buffer.data()));
	CUDA_POST_KERNEL_CHECK;
	caffe_gpu_add_scalar(num_els, lambda, thrust::raw_pointer_cast(buffer.data())); // add scalar lambda for regularization
	caffe_gpu_gemv(CblasNoTrans, num_batch*num_cols, num_rows, static_cast<Dtype>(1.0), thrust::raw_pointer_cast(buffer.data()),
				thrust::raw_pointer_cast(ones.data()), static_cast<Dtype>(0.0), thrust::raw_pointer_cast(col_sums.data()));

	caffe_gpu_scale(col_sums.size(), static_cast<Dtype>(1.0),thrust::raw_pointer_cast(col_sums.data()), sums.mutable_gpu_data());

	caffe_gpu_powx(col_sums.size(), thrust::raw_pointer_cast(col_sums.data()), static_cast<Dtype>(-1.0), thrust::raw_pointer_cast(col_sums.data()));
	caffe_gpu_dgemm(CUBLAS_SIDE_LEFT, num_batch*num_cols, num_rows, thrust::raw_pointer_cast(buffer.data()), thrust::raw_pointer_cast(col_sums.data()), thrust::raw_pointer_cast(buffer.data()));
	Trans_Tensor_kernel<<<CAFFE_GET_BLOCKS(num_els),CAFFE_CUDA_NUM_THREADS>>>(num_els, num_cols, num_rows, thrust::raw_pointer_cast(buffer.data()), output.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void SinkhornNormLayer<Dtype>::RowNorm_GPU(const int num_rows, const int num_cols, const Dtype lambda, Blob<Dtype>& input, Blob<Dtype>& sums, Blob<Dtype>& output){

	// dimensions and aux vectors
	const SinkhornNormParameter& param = this->layer_param_.sinkhorn_norm_param();
	int num_els =  input.count();
	int num_batch = input.shape(0);

	thrust::device_vector<Dtype> ones(num_cols, 1);
	thrust::device_vector<Dtype> row_sums(num_batch*num_rows, 0);

	//Add lambda to each entry for regularization
	caffe_gpu_scale(num_els, static_cast<Dtype>(1.0), input.gpu_data(), output.mutable_gpu_data());
	caffe_gpu_add_scalar(output.count(), lambda, output.mutable_gpu_data());

	// compute row sums - see the input blob 256 x 16 as a list of lines (1024x4)
	caffe_gpu_gemv(CblasNoTrans, num_batch*num_rows, num_cols, static_cast<Dtype>(1.0), output.gpu_data(),
	thrust::raw_pointer_cast(ones.data()), static_cast<Dtype>(0.0), thrust::raw_pointer_cast(row_sums.data()));
	caffe_gpu_scale(row_sums.size(), static_cast<Dtype>(1.0),thrust::raw_pointer_cast(row_sums.data()), sums.mutable_gpu_data());

	// Normalize each entry by sum of rows
	caffe_gpu_powx(row_sums.size(), thrust::raw_pointer_cast(row_sums.data()), static_cast<Dtype>(-1.0), thrust::raw_pointer_cast(row_sums.data()));
	caffe_gpu_dgemm(CUBLAS_SIDE_LEFT, num_batch*num_rows, num_cols, output.gpu_data(), thrust::raw_pointer_cast(row_sums.data()), output.mutable_gpu_data());

}



template <typename Dtype>
void SinkhornNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	// parameters
	const SinkhornNormParameter& param = this->layer_param_.sinkhorn_norm_param();
	bool is_row = param.is_row_step();
	int num_rows = param.num_rows();
	int num_cols = param.num_cols();
	Dtype lambda = static_cast<Dtype>(param.lambda());

	if(is_row){
		RowNorm_GPU(num_rows, num_cols, lambda, (*bottom[0]), sum_norms_, (*top[0]));
	}else{
		ColNorm_GPU(num_rows, num_cols, lambda, (*bottom[0]), sum_norms_, (*top[0]));
	}

}

template <typename Dtype>
void SinkhornNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }

	const SinkhornNormParameter& param = this->layer_param_.sinkhorn_norm_param();
	bool is_row = param.is_row_step();
	int num_els = bottom[0]->count();
	int num_rows = param.num_rows();
	int num_cols = param.num_cols();
	Dtype lambda = static_cast<Dtype>(param.lambda());

	if(is_row){
		RowNorm_Grad_kernel<<<CAFFE_GET_BLOCKS(num_els), CAFFE_CUDA_NUM_THREADS>>>(
				num_els, num_rows, num_cols, lambda, bottom[0]->gpu_data(), sum_norms_.gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		CUDA_POST_KERNEL_CHECK;
	}else{
		ColNorm_Grad_kernel<<<CAFFE_GET_BLOCKS(num_els), CAFFE_CUDA_NUM_THREADS>>>(
				num_els, num_rows, num_cols, lambda, bottom[0]->gpu_data(), sum_norms_.gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		CUDA_POST_KERNEL_CHECK;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SinkhornNormLayer);


}  // namespace caffe

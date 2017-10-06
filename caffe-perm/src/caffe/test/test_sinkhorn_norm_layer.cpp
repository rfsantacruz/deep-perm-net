#include <algorithm>
#include <vector>
#include <boost/assign/list_of.hpp>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sinkhorn_norm_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SinkhornNormLayerTest : public GPUDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	SinkhornNormLayerTest()
: blob_bottom_(new Blob<Dtype>()),
  blob_top_(new Blob<Dtype>()) {
		Caffe::set_random_seed(1701);
		// fill the values
		//FillerParameter filler_param;
		//GaussianFiller<Dtype> filler(filler_param);
		//filler.Fill(this->blob_bottom_);
		vector<int> init_shape  = boost::assign::list_of(10)(16);
		blob_bottom_->Reshape(init_shape);
		caffe_gpu_rng_uniform(blob_bottom_->count(), static_cast<Dtype>(0.0), static_cast<Dtype>(1.0), blob_bottom_->mutable_gpu_data());
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
	}
	virtual ~SinkhornNormLayerTest() { delete blob_bottom_; delete blob_top_; }

	void TestForward(int num_rows, int num_cols, bool is_row_step, float lambda) {
		LayerParameter layer_param;
		layer_param.mutable_sinkhorn_norm_param()->set_num_rows(num_rows);
		layer_param.mutable_sinkhorn_norm_param()->set_num_cols(num_cols);
		layer_param.mutable_sinkhorn_norm_param()->set_is_row_step(is_row_step);
		layer_param.mutable_sinkhorn_norm_param()->set_lambda(lambda);
		SinkhornNormLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		if(layer_param.sinkhorn_norm_param().is_row_step()){
			// check sum and values
			for (int b = 0; b < this->blob_bottom_->shape(0); ++b){
				for( int r = 0; r < num_rows; r++){
					// test if line sum to one
					Dtype sum = 0.0;
					Dtype prev_sum = 0.0;
					for(int c = 0; c< num_cols; c++){
						sum += this->blob_top_->data_at(b, (r*num_cols) + c, 0, 0);
						prev_sum += this->blob_bottom_->data_at(b, (r*num_cols) + c, 0, 0) + lambda;
					}
					EXPECT_NEAR(1.0, sum, 1e-4);
					//test exact values
					for(int c = 0; c< num_cols; c++){
						Dtype expect_value = (this->blob_bottom_->data_at(b, (r*num_cols) + c, 0, 0) + lambda) / ((float)prev_sum);
						Dtype comp_value = this->blob_top_->data_at(b, (r*num_cols) + c, 0, 0);
						EXPECT_NEAR(expect_value, comp_value, 1e-4);
					}
				}
			}
		}else{
			// check sum and values
			for (int b = 0; b < this->blob_bottom_->shape(0); ++b){
				for( int c = 0; c < num_cols; c++){
					// test if line sum to one
					Dtype sum = 0.0;
					Dtype prev_sum = 0.0;
					for(int r = 0; r< num_rows; r++){
						sum += this->blob_top_->data_at(b, (r*num_cols) + c, 0, 0);
						prev_sum += this->blob_bottom_->data_at(b, (r*num_cols) + c, 0, 0) + lambda;
					}
					EXPECT_NEAR(1.0, sum, 1e-4);
					//test exact values
					for(int r = 0; r< num_rows; r++){
						Dtype expect_value = (this->blob_bottom_->data_at(b, (r*num_cols) + c, 0, 0) + lambda)/ ((float)prev_sum);
						Dtype comp_value = this->blob_top_->data_at(b, (r*num_cols) + c, 0, 0);
						EXPECT_NEAR(expect_value, comp_value, 1e-4);
					}
				}
			}
		}
	}


	void TestBackward(int num_rows, int num_cols, bool is_row_step, float lambda) {
		LayerParameter layer_param;
		layer_param.mutable_sinkhorn_norm_param()->set_num_rows(num_rows);
		layer_param.mutable_sinkhorn_norm_param()->set_num_cols(num_cols);
		layer_param.mutable_sinkhorn_norm_param()->set_is_row_step(is_row_step);
		layer_param.mutable_sinkhorn_norm_param()->set_lambda(lambda);
		SinkhornNormLayer<Dtype> layer(layer_param);

		GradientChecker<Dtype> checker(1e-4, 1e-2);
		  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		      this->blob_top_vec_);
	}

	Blob<Dtype>* const blob_bottom_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SinkhornNormLayerTest, TestDtypesAndDevices);

TYPED_TEST(SinkhornNormLayerTest, TestSinkhornForwardRow) {
	typedef typename TypeParam::Dtype Dtype;
	int num_rows = 4;
	int num_cols = 4;
	bool is_row_step = true;
	float lambda = 0.1;
	this->TestForward(num_rows, num_cols, is_row_step, lambda);
}

TYPED_TEST(SinkhornNormLayerTest, TestSinkhornForwardCol) {
	typedef typename TypeParam::Dtype Dtype;
	int num_rows = 4;
	int num_cols = 4;
	bool is_row_step = false;
	float lambda = 0.1;
	this->TestForward(num_rows, num_cols, is_row_step, lambda);
}

TYPED_TEST(SinkhornNormLayerTest, TestSinkhornGradientRow) {
	typedef typename TypeParam::Dtype Dtype;
	int num_rows = 4;
	int num_cols = 4;
	bool is_row_step = true;
	float lambda = 0.1;
	this->TestBackward(num_rows, num_cols, is_row_step, lambda);
}

TYPED_TEST(SinkhornNormLayerTest, TestSinkhornGradientCol) {
	typedef typename TypeParam::Dtype Dtype;
	int num_rows = 4;
	int num_cols = 4;
	bool is_row_step = false;
	float lambda = 0.1;
	this->TestBackward(num_rows, num_cols, is_row_step, lambda);
}

}  // namespace caffe

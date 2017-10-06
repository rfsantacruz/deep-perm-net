#include <algorithm>
#include <vector>
#include <boost/assign/list_of.hpp>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sequence_fusion_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SequenceFusionTest : public GPUDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	SequenceFusionTest()
: blob_bottom_0_(new Blob<Dtype>()), blob_bottom_1_(new Blob<Dtype>()),
  blob_bottom_2_(new Blob<Dtype>()), blob_bottom_3_(new Blob<Dtype>()),
  blob_top_(new Blob<Dtype>()) {
		Caffe::set_random_seed(1701);
		vector<int> init_shape  = boost::assign::list_of(10)(16);
		blob_bottom_0_->Reshape(init_shape); caffe_gpu_set(blob_bottom_0_->count(), static_cast<Dtype>(1.0), blob_bottom_0_->mutable_gpu_data());
		blob_bottom_1_->Reshape(init_shape); caffe_gpu_set(blob_bottom_1_->count(), static_cast<Dtype>(2.0), blob_bottom_1_->mutable_gpu_data());
		blob_bottom_2_->Reshape(init_shape); caffe_gpu_set(blob_bottom_2_->count(), static_cast<Dtype>(3.0), blob_bottom_2_->mutable_gpu_data());
		blob_bottom_3_->Reshape(init_shape); caffe_gpu_set(blob_bottom_3_->count(), static_cast<Dtype>(4.0), blob_bottom_3_->mutable_gpu_data());
		blob_bottom_vec_.push_back(blob_bottom_0_);blob_bottom_vec_.push_back(blob_bottom_1_);blob_bottom_vec_.push_back(blob_bottom_2_);blob_bottom_vec_.push_back(blob_bottom_3_);
		blob_top_vec_.push_back(blob_top_);
	}
	virtual ~SequenceFusionTest() { delete blob_bottom_0_; delete blob_bottom_1_; delete blob_bottom_2_; delete blob_bottom_3_; delete blob_top_; }

	void TestForward() {
		LayerParameter layer_param;
		SequenceFusionLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		// check output dimensions
		EXPECT_EQ(blob_top_vec_.size(), 1);
		EXPECT_EQ(this->blob_top_->shape(0), 10);
		EXPECT_EQ(this->blob_top_->shape(1), 16);

		for (int out_b = 0; out_b < this->blob_top_vec_[0]->shape(0); ++out_b) {
			for (int out_f = 0; out_f < this->blob_top_vec_[0]->shape(1); ++out_f) {
				Dtype exp_value = 10.0; //precumputed
				EXPECT_EQ(this->blob_top_->data_at(out_b, out_f, 0, 0), 10);
			}
		}
	}

	void TestBackward() {
		LayerParameter layer_param;
		SequenceFusionLayer<Dtype> layer(layer_param);

		GradientChecker<Dtype> checker(1e-4, 1e-2);
		  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		      this->blob_top_vec_);
	}

	Blob<Dtype>* const blob_bottom_0_;
	Blob<Dtype>* const blob_bottom_1_;
	Blob<Dtype>* const blob_bottom_2_;
	Blob<Dtype>* const blob_bottom_3_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SequenceFusionTest, TestDtypesAndDevices);

TYPED_TEST(SequenceFusionTest, TestSequenceFusionForward) {
	typedef typename TypeParam::Dtype Dtype;
	this->TestForward();
}

TYPED_TEST(SequenceFusionTest, TestSequenceFusionGradient) {
	typedef typename TypeParam::Dtype Dtype;
	this->TestBackward();
}

}  // namespace caffe

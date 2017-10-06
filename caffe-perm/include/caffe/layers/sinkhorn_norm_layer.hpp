#ifndef CAFFE_SINKHORN_NORM_LAYER_HPP_
#define CAFFE_SINKHORN_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SinkhornNormLayer : public Layer<Dtype> {
 public:
  explicit SinkhornNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SinkhornNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sum_norms_;

 private:
  void RowNorm_GPU(const int num_rows, const int num_cols, const Dtype lambda, Blob<Dtype>& input, Blob<Dtype>& row_sums, Blob<Dtype>& output);
  void ColNorm_GPU(const int num_rows, const int num_cols, const Dtype lambda ,Blob<Dtype>& input, Blob<Dtype>& col_sums, Blob<Dtype>& output);

};

}  // namespace caffe

#endif  // CAFFE_SINKHORN_NORM_LAYER_HPP_

#ifndef CAFFE_BILEVELOPT_NORM_LAYER_HPP_
#define CAFFE_BILEVELOPT_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "gurobi_c++.h"

namespace caffe {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BileveloptNormLayer : public Layer<Dtype> {
 public:
  explicit BileveloptNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BileveloptNorm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual ~BileveloptNormLayer() { delete[] vars; delete[] constrs; delete model; delete env;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> A, b, l;
  GRBModel* model;
  GRBEnv* env;
  GRBVar* vars;
  GRBConstr* constrs;

};

}  // namespace caffe

#endif  // CAFFE_BILEVELOPT_NORM_LAYER_HPP_

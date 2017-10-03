// ------------------------------------------------------------------
// RoiAlignmentLayer written by Thanh-Toan Do
// Only a single value at each bin center is interpolated
// ------------------------------------------------------------------

#ifndef CAFFE_ROI_ALIGNMENT_LAYERS_HPP_
#define CAFFE_ROI_ALIGNMENT_LAYERS_HPP_

//#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/* ROIAlignmentLayer - Region of Interest Alignment Layer
*/

template <typename Dtype>
class ROIAlignmentLayer : public Layer<Dtype> {
 public:
  explicit ROIAlignmentLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIAlignment"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_; // height of conv5_3 (vgg16)
  int width_;  // width of conv5_3
  int pooled_height_; //7
  int pooled_width_;  //7
  Dtype spatial_scale_;
//  Blob<int> max_idx_;

  Blob<int> max_idx_topleft;
  Blob<int> max_idx_topright;
  Blob<int> max_idx_bottomleft;
  Blob<int> max_idx_bottomright;
  Blob<float> dh_ratio;
  Blob<float> dw_ratio;
};

}  // namespace caffe

#endif

// ------------------------------------------------------------------
// RoiAlignmentLayer written by Thanh-Toan Do
// Only a single value at each bin center is interpolated
// ------------------------------------------------------------------

#include <cfloat>
#include <stdio.h>
#include <math.h>
#include <float.h>

//#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/roi_alignment2_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignment2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  ROIAlignment2Parameter roi_align2_param = this->layer_param_.roi_alignment2_param();
  CHECK_GT(roi_align2_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align2_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align2_param.pooled_h(); //7
  pooled_width_ = roi_align2_param.pooled_w();  //7
  spatial_scale_ = roi_align2_param.spatial_scale(); // 1/16
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignment2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  channels_ = bottom[0]->channels(); // bottom[0]: conv5_3 (vgg16), // bottom[1]: rois
  height_ = bottom[0]->height(); // height of conv5_3
  width_ = bottom[0]->width();   // width of conv5_3
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // N x C x 7 x 7 where N is number of ROIS
//  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_x.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // Blob<int> max_idx_;
  max_idx_y.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void ROIAlignment2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	NOT_IMPLEMENTED;
}


template <typename Dtype>
void ROIAlignment2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignment2Layer);
#endif

INSTANTIATE_CLASS(ROIAlignment2Layer);
REGISTER_LAYER_CLASS(ROIAlignment2);

}  // namespace caffe

// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>
#include <stdio.h>
#include <math.h>
#include <float.h>

//#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/roi_alignment_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  ROIAlignmentParameter roi_align_param = this->layer_param_.roi_alignment_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h(); 
  pooled_width_ = roi_align_param.pooled_w();  
  spatial_scale_ = roi_align_param.spatial_scale(); 
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  channels_ = bottom[0]->channels(); 
  height_ = bottom[0]->height(); 
  width_ = bottom[0]->width();   
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); 
//  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); 
  //////////////////////////////////////////////////////////////////////////////
  max_idx_topleft.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_topright.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_bottomleft.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  max_idx_bottomright.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  dh_ratio.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  dw_ratio.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); //
  /////////////////////////////////////////////////////////////////////////////
}


template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data(); 
  const Dtype* bottom_rois = bottom[1]->cpu_data(); 

  int num_rois = bottom[1]->num(); 
  int batch_size = bottom[0]->num(); 
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
//int* argmax_data = max_idx_.mutable_cpu_data();
//caffe_set(top_count, -1, argmax_data);
  /////////////////////////////////////////////////////////////////
  int* argmax_data_topleft = max_idx_topleft.mutable_cpu_data();
  int* argmax_data_topright = max_idx_topright.mutable_cpu_data();
  int* argmax_data_bottomleft = max_idx_bottomleft.mutable_cpu_data();
  int* argmax_data_bottomright = max_idx_bottomright.mutable_cpu_data();
  float* dh_ratio_data = dh_ratio.mutable_cpu_data();
  float* dw_ratio_data = dw_ratio.mutable_cpu_data();

  caffe_set(top_count, -1, argmax_data_topleft);
  caffe_set(top_count, -1, argmax_data_topright);
  caffe_set(top_count, -1, argmax_data_bottomleft);
  caffe_set(top_count, -1, argmax_data_bottomright);
  caffe_set(top_count, (float)(0), dh_ratio_data);
  caffe_set(top_count, (float)(0), dw_ratio_data);
  /////////////////////////////////////////////////////////////////
  for (int n = 0; n < num_rois; ++n) 
  {
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = (bottom_rois[1] * spatial_scale_); 
    float roi_start_h = (bottom_rois[2] * spatial_scale_); 
    float roi_end_w = (bottom_rois[3] * spatial_scale_);  
    float roi_end_h = (bottom_rois[4] * spatial_scale_);  
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1); 
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1); 
    float bin_size_h = (roi_height) / ((float)(pooled_height_)); 
    float bin_size_w = (roi_width) / ((float)(pooled_width_)); 

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c)  
    {
      for (int ph = 0; ph < pooled_height_; ++ph) 
      {
        for (int pw = 0; pw < pooled_width_; ++pw)  
        {
          float hstart = ((float)(ph))* bin_size_h;
          float wstart = ((float)(pw))* bin_size_w;
          float hend = ((float)(ph + 1))* bin_size_h;
          float wend = ((float)(pw + 1))* bin_size_w;

          hstart = fminf(fmaxf(hstart + roi_start_h, 0), height_ -1);
          hend = fminf(fmaxf(hend + roi_start_h, 0), height_ -1);
          wstart = fminf(fmaxf(wstart + roi_start_w, 0), width_ -1);
          wend = fminf(fmaxf(wend + roi_start_w, 0), width_ -1);
          bool is_empty = (hend < hstart) || (wend < wstart);

          const int pool_index = ph * pooled_width_ + pw; 
          if (is_empty)
          {
            top_data[pool_index] = 0;
//            argmax_data[pool_index] = -1;
            argmax_data_topleft[pool_index] = -1;
            argmax_data_topright[pool_index] = -1;
            argmax_data_bottomleft[pool_index] = -1;
            argmax_data_bottomright[pool_index] = -1;
            dh_ratio_data[pool_index] = 0;
            dw_ratio_data[pool_index] = 0;
          }
          else
          {
			  float centerx = (wstart + wend)/2;
			  float centery = (hstart + hend)/2;

			  int cy_top = 	min(  max(  static_cast<int>(floor(centery)), 0)  ,height_-1);
			  int cy_bottom = min(  max(  static_cast<int>(ceil(centery)), 0), height_-1);
			  int cx_left = min(  max(  static_cast<int>(floor(centerx)), 0), width_-1);
			  int cx_right = min(  max(  static_cast<int>(ceil(centerx)), 0), width_-1);

			  int topleft = cy_top * width_ + cx_left;
			  int topright = cy_top * width_ + cx_right;
			  int bottomleft = cy_bottom * width_ + cx_left;
			  int bottomright = cy_bottom * width_ + cx_right;

			  float y_ratio =  centery - (float)(cy_top); 
			  float x_ratio =  centerx - (float)(cx_left); 

			  top_data[pool_index] = batch_data[topleft] * (1-y_ratio) * (1-x_ratio)
								  +  batch_data[topright] * (1-y_ratio) * (x_ratio)
								  +  batch_data[bottomleft] * (y_ratio) * (1 - x_ratio)
								  +  batch_data[bottomright] * (y_ratio) * (x_ratio);

			  argmax_data_topleft[pool_index] = topleft; 
			  argmax_data_topright[pool_index] = topright;
			  argmax_data_bottomleft[pool_index] = bottomleft;
			  argmax_data_bottomright[pool_index] = bottomright; 
			  dh_ratio_data[pool_index] = y_ratio;
			  dw_ratio_data[pool_index] = x_ratio;
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
//      argmax_data += max_idx_.offset(0, 1);
      argmax_data_topleft += max_idx_topleft.offset(0, 1);
      argmax_data_topright += max_idx_topright.offset(0, 1);
      argmax_data_bottomleft += max_idx_bottomleft.offset(0, 1);
      argmax_data_bottomright += max_idx_bottomright.offset(0, 1);
      dh_ratio_data += dh_ratio.offset(0, 1);
      dw_ratio_data += dw_ratio.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}


template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignmentLayer);
#endif

INSTANTIATE_CLASS(ROIAlignmentLayer);
REGISTER_LAYER_CLASS(ROIAlignment);

}  // namespace caffe

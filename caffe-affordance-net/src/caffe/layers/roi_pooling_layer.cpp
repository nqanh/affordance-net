// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h(); //7
  pooled_width_ = roi_pool_param.pooled_w();  //7
  spatial_scale_ = roi_pool_param.spatial_scale(); //16
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels(); // bottom[0]: conv5_3, // bottom[1]: rois
  height_ = bottom[0]->height(); // height of conv5_3
  width_ = bottom[0]->width();   // width of conv5_3
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // N x C x 7 x 7
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_); // Blob<int> max_idx_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); //conv5_3
  const Dtype* bottom_rois = bottom[1]->cpu_data(); // rois

  int num_rois = bottom[1]->num(); // Number of ROIs
  int batch_size = bottom[0]->num(); // number of image = 1
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) //scan each roi
  {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_); // x1*1/16. Because (x1,y1,x2,y2) is coordinate in input image, divide by 16 --> coordinate conv5_3
    int roi_start_h = round(bottom_rois[2] * spatial_scale_); // y1*1/16
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);  // x2*1/16
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);  // y2*1/16
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1); // roi_height: height of roi (w.r.t. conv5_3 size)
    int roi_width = max(roi_end_w - roi_start_w + 1, 1); // roi_width: width of roi (w.r.t. conv5_3 size)
    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_); // bin_size_h = roi_height/7 = height stride (w.r.t. conv5_3 size)
    const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_); // bin_size_w = roi_width/7 = width stride (w.r.t. conv5_3 size)

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c)  // for each channel in conv5_3
    {
      // scan each bin in 7x7 grid
      for (int ph = 0; ph < pooled_height_; ++ph) // pooled_height_ = 7
      {
        for (int pw = 0; pw < pooled_width_; ++pw)  // pooled_width_ = 7
        {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          // (hstart, wstart), (hend, wend): sub window in conv5_3 in which max value is got
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)* bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)* bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)* bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)* bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_); // hstart = hstart + roi_start_h: coordinate of subwindow in conv5_3
          hend = min(max(hend + roi_start_h, 0), height_); // hend = hend + roi_start_h: coordinate of subwindow in conv5_3
          wstart = min(max(wstart + roi_start_w, 0), width_); // wstart = wstart + roi_start_w: coordinate of subwindow in conv5_3
          wend = min(max(wend + roi_start_w, 0), width_); // wend = wend + roi_start_w: wend + roi_start_w

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw; // bin id (in 7x7 grid)
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }
          // find max value in sub window (wstart, hstart, wstart, wend). (wstart, hstart, wstart, wend) is coordinate in conv5_3
          for (int h = hstart; h < hend; ++h)
          {
            for (int w = wstart; w < wend; ++w)
            {
              const int index = h * width_ + w; // index in conv5_3
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index; // store the index where max value is got; this index coordinate is in conv5_3
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

}  // namespace caffe

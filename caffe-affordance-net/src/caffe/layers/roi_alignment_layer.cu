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

namespace caffe {

template <typename Dtype>
//__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
//    const Dtype spatial_scale, const int channels, const int height,
//    const int width, const int pooled_height, const int pooled_width,
//    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data)

__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data,
    int* argmax_data_topleft, int* argmax_data_topright,
    int* argmax_data_bottomleft, int* argmax_data_bottomright,
    float* dh_ratio_data, float* dw_ratio_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = (float)(bottom_rois[1] * spatial_scale); 
    float roi_start_h = (float)(bottom_rois[2] * spatial_scale);
    float roi_end_w = (float)(bottom_rois[3] * spatial_scale);
    float roi_end_h = (float)(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    float roi_width =  fmaxf(roi_end_w - roi_start_w + 1, 1);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
    float bin_size_h = (roi_height) / ((float)(pooled_height));
    float bin_size_w = (roi_width) / ((float)(pooled_width));

    float hstart = ((float)(ph)) * bin_size_h;
    float wstart = ((float)(pw)) * bin_size_w;
    float hend = ((float)(ph + 1)) * bin_size_h;
    float wend = ((float)(pw + 1)) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = fminf(fmaxf(hstart + roi_start_h, 0), height-1);
    hend = fminf(fmaxf(hend + roi_start_h, 0), height-1);
    wstart = fminf(fmaxf(wstart + roi_start_w, 0), width-1);
    wend = fminf(fmaxf(wend + roi_start_w, 0), width-1);

    bool is_empty = (hend < hstart) || (wend < wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;

    if (is_empty)
    {
    	top_data[index] = maxval;
//    	argmax_data[index] = maxidx;
    	argmax_data_topleft[index] = maxidx;
    	argmax_data_topright[index] = maxidx;
    	argmax_data_bottomleft[index] = maxidx;
    	argmax_data_bottomright[index] = maxidx;
    	dh_ratio_data[index] = 0.0;
    	dw_ratio_data[index] = 0.0;
    }
    else
    {
    	float centerx = (float)(wstart + wend)/2.0;
    	float centery = (float)(hstart + hend)/2.0;

		int cy_top = 	static_cast<int>(floor(centery));
		int cy_bottom = static_cast<int>(ceil(centery));
		int cx_left = 	static_cast<int>(floor(centerx));
		int cx_right = 	static_cast<int>(ceil(centerx));

		cy_top = 	min(max(cy_top, 0), height-1);
		cy_bottom = min(max(cy_bottom, 0), height-1);
		cx_left = 	min(max(cx_left, 0), width-1);
		cx_right = 	min(max(cx_right, 0), width-1);

		int topleft = cy_top * width + cx_left;
	    int topright = cy_top * width + cx_right;
		int bottomleft = cy_bottom * width + cx_left;
		int bottomright = cy_bottom * width + cx_right;

		// bilinear interpolation
		float y_ratio =  centery - (float)(cy_top); 
		float x_ratio =  centerx - (float)(cx_left); 

		maxval  =  bottom_data[topleft] * (1-y_ratio) * (1-x_ratio)
				+  bottom_data[topright] * (1-y_ratio) * (x_ratio)
				+  bottom_data[bottomleft] * (y_ratio) * (1 - x_ratio)
				+  bottom_data[bottomright] * (y_ratio) * (x_ratio);

    	top_data[index] = maxval;
//    	argmax_data[index] = maxidx;
    	argmax_data_topleft[index] = topleft;
    	argmax_data_topright[index] = topright;
    	argmax_data_bottomleft[index] = bottomleft;
    	argmax_data_bottomright[index] = bottomright;
    	dh_ratio_data[index] = y_ratio;
    	dw_ratio_data[index] = x_ratio;
    }
  }
}

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
//  int* argmax_data = max_idx_.mutable_gpu_data();
  /////////////////////////////////////////////////
  int* argmax_data_topleft = max_idx_topleft.mutable_gpu_data();
  int* argmax_data_topright = max_idx_topright.mutable_gpu_data();
  int* argmax_data_bottomleft = max_idx_bottomleft.mutable_gpu_data();
  int* argmax_data_bottomright = max_idx_bottomright.mutable_gpu_data();
  float* dh_ratio_data = dh_ratio.mutable_gpu_data();
  float* dw_ratio_data = dw_ratio.mutable_gpu_data();
  /////////////////////////////////////////////////
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
//  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//      count, bottom_data, spatial_scale_, channels_, height_, width_,
//      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data,
      argmax_data_topleft, argmax_data_topright,
      argmax_data_bottomleft, argmax_data_bottomright,
      dh_ratio_data, dw_ratio_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
//__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
//    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
//    const int channels, const int height, const int width,
//    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
//    const Dtype* bottom_rois)
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data_topleft, const int* argmax_data_topright,
    const int* argmax_data_bottomleft, const int* argmax_data_bottomright,
    const float* dh_ratio_data, const float* dw_ratio_data,
    const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    // (n, c, h, w) coords in bottom data (in conv5_3)
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0.;

    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n)
    {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind){
        continue;
      }

      float roi_start_w = (float)(offset_bottom_rois[1] * spatial_scale);
      float roi_start_h = (float)(offset_bottom_rois[2] * spatial_scale);
      float roi_end_w = (float)(offset_bottom_rois[3] * spatial_scale);
      float roi_end_h = (float)(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      //const bool in_roi = (w >= roi_start_w && w <= roi_end_w && h >= roi_start_h && h <= roi_end_h);
      const bool in_roi = (w >= roi_start_w-1 && w <= roi_end_w+1 && h >= roi_start_h-1 && h <= roi_end_h+1); 
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
//      const int* offset_argmax_data = argmax_data + offset;

      ///////////////////////////////////////////////////////
      const int* offset_argmax_data_topright = argmax_data_topright + offset;
      const int* offset_argmax_data_topleft =  argmax_data_topleft + offset;
      const int* offset_argmax_data_bottomleft = argmax_data_bottomleft + offset;
      const int* offset_argmax_data_bottomright = argmax_data_bottomright + offset;

      const float* offset_dh_ratio_data = dh_ratio_data + offset;
      const float* offset_dw_ratio_data = dw_ratio_data + offset;
      ///////////////////////////////////////////////////////

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
      float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);

      float bin_size_h = (roi_height) / ((float)(pooled_height));
      float bin_size_w = (roi_width) / ((float)(pooled_width));

      int phstart = floor(((float)h - roi_start_h) / bin_size_h);
      int phend = ceil(((float)h - roi_start_h) / bin_size_h);
      int pwstart = floor(((float)w - roi_start_w) / bin_size_w);
      int pwend = ceil(((float)w - roi_start_w) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      phstart = 0;
      phend = pooled_height;
      pwstart = 0;
      pwend = pooled_width;

      for (int ph = phstart; ph < phend; ++ph)
      {
        for (int pw = pwstart; pw < pwend; ++pw)
        {
        	int topright = offset_argmax_data_topright[ph * pooled_width + pw];
        	int topleft = offset_argmax_data_topleft[ph * pooled_width + pw];
        	int bottomleft = offset_argmax_data_bottomleft[ph * pooled_width + pw];
        	int bottomright = offset_argmax_data_bottomright[ph * pooled_width + pw];

        	float y_ratio = offset_dh_ratio_data[ph * pooled_width + pw];
        	float x_ratio = offset_dw_ratio_data[ph * pooled_width + pw];

        	if (topleft == (h * width + w))
			{
        		gradient += (offset_top_diff[ph * pooled_width + pw] * (1. - y_ratio)*(1. - x_ratio));
        	}
        	if (topright == (h * width + w))
			{
        		gradient += (offset_top_diff[ph * pooled_width + pw]*(1. -y_ratio)*(x_ratio));
        	}
        	if (bottomleft == (h * width + w))
			{
        		gradient += (offset_top_diff[ph * pooled_width + pw]* (y_ratio) * (1. - x_ratio));
        	}
        	if (bottomright == (h * width + w))
			{
        		gradient += (offset_top_diff[ph * pooled_width + pw]*(y_ratio) * (x_ratio));
        	}
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignmentLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
//  const int* argmax_data = max_idx_.gpu_data();
  /////////////////////////////////////////////////////////////////
  const int* argmax_data_topleft = max_idx_topleft.gpu_data();
  const int* argmax_data_topright = max_idx_topright.gpu_data();
  const int* argmax_data_bottomleft = max_idx_bottomleft.gpu_data();
  const int* argmax_data_bottomright = max_idx_bottomright.gpu_data();
  const float* dh_ratio_data = dh_ratio.gpu_data();
  const float* dw_ratio_data = dw_ratio.gpu_data();
  ////////////////////////////////////////////////////////////////
  // NOLINT_NEXT_LINE(whitespace/operators)
//  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
//      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff,
      argmax_data_topleft, argmax_data_topright,
      argmax_data_bottomleft, argmax_data_bottomright,
      dh_ratio_data, dw_ratio_data,
      top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignmentLayer);

}  // namespace caffe

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps

import cv2
import os.path as osp
import cPickle
DEBUG = False

class PredToProposalLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        #self._num_classes = layer_params['num_classes']
        self._max_per_image = layer_params['max_per_image']
        self._thresh = layer_params['thresh']
        # rois_for_mask (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # rois_class_score
        top[1].reshape(1, 1)
        # rois_class_ind
        top[2].reshape(1, 1)
        # rois_final
        top[3].reshape(1, 5)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        rois = bottom[0].data

        # bbox_pred
        box_deltas = bottom[1].data
        # class score
        scores = bottom[2].data
        # image info
        im_info = bottom[3].data
        im_scale = im_info[0][2]

        boxes_0 = rois[:, 1:5] / im_scale
        pred_boxes = bbox_transform_inv(boxes_0, box_deltas)
        im_shape = [im_info[0][0], im_info[0][1]]/im_scale 
        boxes = clip_boxes(pred_boxes, im_shape)

        max_per_image = self._max_per_image
        thresh = self._thresh
        num_classes = scores.shape[1]
        i = 0 
        num_images = 1
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]

        for j in xrange(1, num_classes):

            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4] 

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        num_boxes = 0
        for j in xrange(1, num_classes):
            num_boxes = num_boxes + all_boxes[j][i].shape[0]

        num_boxes = max(num_boxes,1)

        rois_for_mask = np.zeros((num_boxes, 5), dtype=np.float32)
        rois_class_score = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_class_ind = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_final = np.zeros((num_boxes, 5), dtype=np.float32)

        count = 0
        for j in xrange(1, num_classes):
            all_boxes_j = all_boxes[j][i]
            c = all_boxes_j.shape[0]
            if c > 0:
                coors = all_boxes_j[:, 0:4]
                cl_scores = all_boxes_j[:, 4:5]

                rois_for_mask[count:count+c, 1:5] = coors*im_scale
                rois_final[count:count+c, 1:5] = coors
                rois_class_score[count:count+c, 0:1] = cl_scores
                rois_class_ind[count:count+c, 0:1] = np.tile(j, [c, 1])
                count = count + c

        top[0].reshape(*rois_for_mask.shape)
        top[0].data[...] = rois_for_mask
        top[1].reshape(*rois_class_score.shape)
        top[1].data[...] = rois_class_score

        # class index
        top[2].reshape(*rois_class_ind.shape)
        top[2].data[...] = rois_class_ind

        # rois_final
        top[3].reshape(*rois_final.shape)
        top[3].data[...] = rois_final

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



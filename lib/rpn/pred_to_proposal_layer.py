# --------------------------------------------------------
# Select top 100 predicted boxes for mask branch
# Written by Thanh-Toan Do
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

        # print ("==================================rois====================")
        # print rois

        ######## bbox_pred
        box_deltas = bottom[1].data
        ########class score
        scores = bottom[2].data
        ########image info
        im_info = bottom[3].data
        im_scale = im_info[0][2]

        # unscale back to raw image space
        boxes_0 = rois[:, 1:5] / im_scale
        pred_boxes = bbox_transform_inv(boxes_0, box_deltas)
        im_shape = [im_info[0][0], im_info[0][1]]/im_scale #original size of input image
        boxes = clip_boxes(pred_boxes, im_shape) #clip predicted box using original input size

        # print("=========================rois from rpn.proposal_layer")
        # print("=========================shape: " + str(rois.shape))
        # print rois

        # print("=========================rois from rpn.proposal_layer")
        # print("=========================shape: " + str(boxes.shape))
        # print boxes

        max_per_image = self._max_per_image
        thresh = self._thresh
        num_classes = scores.shape[1]
        i = 0 #only support single image
        num_images = 1
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(num_classes)]

        # print ("=========================num_classes: " + str(num_classes))
        # print ("=========================image size: " + str(im_shape))

        ## for each class (ignoring background class)
        for j in xrange(1, num_classes):

            # if j == 23:
            #     print ("=========================scores[:,j]. j = " + str(j))
            #     print scores[:, j]

            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4] #get boxes correspond to class j

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            # print ("===============================size of dets before nms: " + str(cls_dets.shape))
            # cfg.TEST.NMS = 0.3
            keep = nms(cls_dets, cfg.TEST.NMS)
            # print ("===============keep in rpn/pred_to_proposal_layer.py======: " + str(keep))
            cls_dets = cls_dets[keep, :]
            # print ("===============================size of dets after nms: " + str(cls_dets.shape))
            all_boxes[j][i] = cls_dets

            # print ("===================image: " + str(i) + " class: " + str(j))
            # print ("===================shape of all_boxes[j][i]: " + str(all_boxes[j][i].shape))
            # print all_boxes[j][i]

            # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    # print ("===================image: " + str(i) + "class: " + str(j))
                    # print ("===================shape of all_boxes[j][i]: " + str(all_boxes[j][i].shape))

        num_boxes = 0
        for j in xrange(1, num_classes):
            num_boxes = num_boxes + all_boxes[j][i].shape[0]

        # print ("===========num_boxes========:" + str(num_boxes))
        num_boxes = max(num_boxes,1) #tranh loi 'Floating point exception(core dumped)'

        rois_for_mask = np.zeros((num_boxes, 5), dtype=np.float32)
        rois_class_score = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_class_ind = -1*np.ones((num_boxes, 1), dtype=np.float32)
        rois_final = np.zeros((num_boxes, 5), dtype=np.float32)

        count = 0
        for j in xrange(1, num_classes):
            all_boxes_j = all_boxes[j][i] #boxes correspond to class j
            c = all_boxes_j.shape[0]
            if c > 0:
                coors = all_boxes_j[:, 0:4]
                cl_scores = all_boxes_j[:, 4:5]

                rois_for_mask[count:count+c, 1:5] = coors*im_scale # w.r.t big size, e.g., 600x1000
                rois_final[count:count+c, 1:5] = coors # w.r.t. original image size. rois_final same rois_for_mask but with different scale
                rois_class_score[count:count+c, 0:1] = cl_scores
                rois_class_ind[count:count+c, 0:1] = np.tile(j, [c, 1])
                count = count + c

        # print ("===================================rois_for_mask")
        # print ("===================================shape: " + str(rois_for_mask.shape))
        # print rois_for_mask

        # rois_for_mask
        # print ("===========OK or NOT========")
        top[0].reshape(*rois_for_mask.shape)
        top[0].data[...] = rois_for_mask
        # print ("===========OK or NOT========")
        # classification score
        top[1].reshape(*rois_class_score.shape)
        top[1].data[...] = rois_class_score

        # class index
        top[2].reshape(*rois_class_ind.shape)
        top[2].data[...] = rois_class_ind

        # rois_final
        top[3].reshape(*rois_final.shape)
        top[3].data[...] = rois_final

        # print ("=======================number of rois_final:====================" + str(rois_final.shape))

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



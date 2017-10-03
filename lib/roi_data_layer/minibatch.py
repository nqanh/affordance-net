# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob, crop_pad_im_for_blob


print_verbose = 0

# ########################################################
# ## Anh get minibatch - no resize - do crop or padding
# def get_minibatch(roidb, num_classes):
#     """Given a roidb, construct a minibatch sampled from it."""
#     num_images = len(roidb)
#     # Sample random scales to use for each image in this batch
#     random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
#                                     size=num_images)
#     assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
#         'num_images ({}) must divide BATCH_SIZE ({})'. \
#         format(num_images, cfg.TRAIN.BATCH_SIZE)
#     rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
#     fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
# 
#     # Get the input image blob, formatted for caffe
#     im_blob, crop_offsets = _get_image_blob(roidb) #im_blob la image sau khi da crop + padd 
#     #print '============ crop offsets: ', crop_offsets
#     
#     blobs = {'data': im_blob}
# 
#     if cfg.TRAIN.HAS_RPN: #=TRUE neu la end2end (xem file faster_rcnn_end2end.yml)
#         #assert len(im_scales) == 1, "Single batch only"
#         assert len(roidb) == 1, "Single batch only"
#         # gt boxes: (x1, y1, x2, y2, cls)
#         gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
#         gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
#         
#         # get orignal bboxes
#         ori_gt_boxes = roidb[0]['boxes'][gt_inds, :] ## size: [ [xmin, ymin, xmax, ymax] [] ]
#         if print_verbose: print '--- original bboxes: ', ori_gt_boxes
#         new_gt_boxes = _transform_bboxes(ori_gt_boxes, crop_offsets[0])  # convert orignal bbox coordinates to new coordinate at new croped_image
#         # crop_offsets[0] --> Get the only 1 offset in list
#         if print_verbose: print '--- new gt bboxes  : ', new_gt_boxes
#         
#         #gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0] #chuyen toa do cua box theo kich thuoc anh sau khi scale. [gt_inds, :] : vi 1 box la 1 array. boxes la 2D array
#         gt_boxes[:, 0:4] = new_gt_boxes # set gt bboxes based on new bbox
#         gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds] #[gt_inds]: vi for each box, gt_classes la 1 scalar.
#         blobs['gt_boxes'] = gt_boxes #matrix (1,5)
#         
# #         blobs['im_info'] = np.array(
# #             [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],dtype=np.float32) #im_blob.shape[2]= height, im_blob.shape[3] = width
#         
#         blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], crop_offsets[0][0], crop_offsets[0][1] ]],dtype=np.float32) # TODO: CHECK IT #im_blob.shape[2]= height, im_blob.shape[3] = width
#         # crop_offsets[0][0] = start_h; crop_offsets[0][1] = start_w
#         
#         
#         ######################read mask index######################
#         if cfg.TRAIN.MASK_REG:
#             # print "====================Check roidb's fields======================"
#             # print roidb[0]
#             ''' --> in roidb[0] has: 
#             {
#                 'gt_classes': array([ 7, 10,  1, 10,  7, 10], dtype=int32), 
#                 'max_classes': array([ 7, 10,  1, 10,  7, 10]), 
#                 'image': './data/coco/images/train2014/COCO_train2014_000000255576.jpg', 
#                 'bbox_targets': array([[  7.,   0.,   0.,   0.,   0.],
#                [ 10.,   0.,   0.,   0.,   0.],
#                [  1.,   0.,   0.,   0.,   0.],
#                [ 10.,   0.,   0.,   0.,   0.],
#                [  7.,   0.,   0.,   0.,   0.],
#                [ 10.,   0.,   0.,   0.,   0.]], dtype=float32), 
#                'seg_mask_inds': array([[255576,      1],
#                [255576,      2],
#                [255576,      3],
#                [255576,      4],
#                [255576,      5],
#                [255576,      6]], dtype=uint32), 
#                'flipped': False / True. Depend on this image is original or flipped
#                'width': 640, ------> original image size
#                'boxes': array([[114,  99, 541, 386],
#                [253, 122, 263, 147],
#                [472, 170, 518, 212],
#                [142,  89, 167, 152],
#                [ 39, 245,  68, 305],
#                [266, 124, 280, 131]], dtype=uint16), 
#                'max_overlaps': array([ 1.,  1.,  1.,  1.,  1.,  1.], dtype=float32), 
#                'height': 427, ------> original image size
#                'seg_areas': array([ 97571.140625  ,    230.68635559,   1110.08752441,   1616.10449219,
#                  1478.43811035,    109.41735077], dtype=float32), 
#                  'gt_overlaps': <6x81 sparse matrix of type '<type 'numpy.float32'>'
#                 with 6 stored elements in Compressed Sparse Row format>
#             }
#             '''
#             seg_mask_inds = np.empty((len(gt_inds), 2), dtype=np.int32)
#             seg_mask_inds[:, 0:2] = roidb[0]['seg_mask_inds'][gt_inds, :] #each gt_box has 1 correspondence seg_mask_ind
#             blobs['seg_mask_inds'] = seg_mask_inds
# 
#             flipped = np.empty((1,1), dtype=np.bool_)
#             flipped[0, 0] = roidb[0]['flipped']
#             blobs['flipped'] = flipped
# 
#             # print "====================Check blobs's fields======================"
#             # print blobs
#             '''
#             {
#             'gt_boxes': array([[  64.78873444,  200.,  723.94366455,  487.32394409,   37.        ],
#                                 [ 100.        ,    1.40845072,  661.97180176,  307.04226685,    1.        ]], dtype=float32), 
#             'data': array([[[[ 36.01990128,  46.75490189,  48.96990204, ..., -59.49517822,
#               -45.75061035, -64.20959473],
#              [ 41.66989899,  46.97807693,  50.60840225, ..., -60.74946594,
#               -50.01625061, -64.46395111],
#              [ 44.09490204,  46.77865601,  51.63615417, ..., -64.77590179,
#               -55.13316727, -62.00203705],
#              ..., 
#              [ 44.99496841,  55.27793884,  66.12365723, ...,  -0.76256037,
#                -9.7339344 ,   1.87395406],
#              [ 41.4949379 ,  47.3398819 ,  53.23834991, ...,   3.3928864 ,
#                -1.27210546,  10.27186584],
#              [ 33.01990128,  34.71490097,  36.01990128, ...,   2.01990008,
#                 4.05017328,  13.98962593]],]]], dtype=float32), 
#             'seg_mask_inds': array([[555074,      1],   [555074,      2]], dtype=int32), 
#             'im_info': array([[ 600.        ,  901.        ,    1.40845072]], dtype=float32), #height/width of image after scaling.  1.40845072: used scale 
#             'flipped': array([[False]], dtype=bool) # flipped = False --> image nay la original
#            }
#             '''
#         ###########################################################
# 
# ## DISSABLE
# #     else: # not using RPN
# #         # Now, build the region of interest and label blobs
# #         rois_blob = np.zeros((0, 5), dtype=np.float32)
# #         labels_blob = np.zeros((0), dtype=np.float32)
# #         bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
# #         bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
# #         # all_overlaps = []
# #         for im_i in xrange(num_images):
# #             labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
# #                 = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
# #                                num_classes)
# # 
# #             # Add to RoIs blob
# #             rois = _project_im_rois(im_rois, im_scales[im_i])
# #             batch_ind = im_i * np.ones((rois.shape[0], 1))
# #             rois_blob_this_image = np.hstack((batch_ind, rois))
# #             rois_blob = np.vstack((rois_blob, rois_blob_this_image))
# # 
# #             # Add to labels, bbox targets, and bbox loss blobs
# #             labels_blob = np.hstack((labels_blob, labels))
# #             bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
# #             bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
# #             # all_overlaps = np.hstack((all_overlaps, overlaps))
# # 
# #         # For debug visualizations
# #         # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
# # 
# #         blobs['rois'] = rois_blob
# #         blobs['labels'] = labels_blob
# # 
# #         if cfg.TRAIN.BBOX_REG:
# #             blobs['bbox_targets'] = bbox_targets_blob
# #             blobs['bbox_inside_weights'] = bbox_inside_blob
# #             blobs['bbox_outside_weights'] = \
# #                 np.array(bbox_inside_blob > 0).astype(np.float32)
# 
#     return blobs
####################################################################################


def _transform_bboxes(org_boxes, crop_offset):
    '''
    transform original bboxs coordinates to new coordinates at croped_image
    '''
    #print 'org boxes shape: ', org_boxes.shape ## COULD BE MANY BOX
    new_list_bboxes = [] # keep list of new bbox coordinate w.r.t new croped im
    old_list_bboxes = []
    
    #for ob, cp in zip(org_boxes, crop_offsets):
    for ob in org_boxes:
#         al = al.strip() # remove '\n'
#         arr = al.split(' ')
#         
#         xmin = int(arr[1])
#         ymin = int(arr[2])
#         xmax = int(arr[3])
#         ymax = int(arr[4])
        
        xmin = ob[0]
        ymin = ob[1]
        xmax = ob[2]
        ymax = ob[3]
        
        old_box = [xmin, ymin, xmax, ymax]
        old_list_bboxes.append(old_box)
        
        start_h = crop_offset[0]
        start_w = crop_offset[1]
        
        crop_size = cfg.IMG_CROPPAD_SIZE
        
        new_xmin = xmin - start_w ## new_xmin --> new bbox corrdinate in croped_im
        new_ymin = ymin - start_h
        new_xmax = xmax - start_w
        new_ymax = ymax - start_h
        
        # reset coordinate if < 0 --> only get good region
        if new_xmin < 0:
            new_xmin = 0
        if new_ymin < 0:
            new_ymin = 0
            
        if new_xmax < 0:
            new_xmax = 0
        if new_ymax < 0: 
            new_ymax = 0
        
        if new_xmax > crop_size:
            new_xmax = crop_size
        if new_ymax > crop_size:
            new_ymax = crop_size
        
        if new_xmin > crop_size or new_ymin > crop_size:
#             print 'found 1 invalid bbox'
            new_xmin = new_ymin = new_xmax = new_ymax = 0
        
        # check if we have invalid bbox (object is outside from croped_img) --> reset new_box to all 0 = denote invalid box
        if new_ymin == 0 and new_ymax == 0:
#             print 'found 1 invalid bbox'
            new_xmin = new_ymin = new_xmax = new_ymax = 0
            
        if new_xmin == 0 and new_xmax == 0:
            new_xmin = new_ymin = new_xmax = new_ymax = 0
            
        new_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_list_bboxes.append(new_box)
        
    
    #print 'new list bboxes: ', new_list_bboxes
        
## DO NOT USE - WRONG bbox class when reset bbox without reseting object class
#     # reset all zero with valid one 
#     num_bboxes = len(new_list_bboxes) 
#     if num_bboxes > 1:
#         good_box = []
#         for i in range(num_bboxes): # find 1 good box
#             curr_box = new_list_bboxes[i]
#             if np.count_nonzero(curr_box) > 0:
#                 #print 'get good box: ', curr_box
#                 good_box = curr_box
#                 break
#         #print 'find good box: ', good_box
#         if len(good_box) != 0: # good box is not empty
#             for i in range(num_bboxes): # reset to good box
#                 if np.count_nonzero(new_list_bboxes[i]) == 0:
#                     new_list_bboxes[i] = good_box
    
    
    selected_list_bboxes = []
    # remove new box if its area is too small
    for ix, new_box in enumerate(new_list_bboxes):
        #print 'ix: ', ix
        nxmin = new_box[0]
        nymin = new_box[1]
        nxmax = new_box[2]
        nymax = new_box[3]
        
        old_box = old_list_bboxes[ix]
        oxmin = old_box[0]
        oymin = old_box[1]
        oxmax = old_box[2]
        oymax = old_box[3]
        
        narea = (nxmax - nxmin) * (nymax-nymin)
        oarea = (oxmax - oxmin) * (oymax-oymin)
        
        if narea > oarea * cfg.GOOD_BOX_PERCENTAGE:  # new area is big enough ~ (85 --> 100%) old area
            selected_list_bboxes.append(new_box)
        else:
            selected_list_bboxes.append([0, 0, 0, 0]) # discard this box
        
    #print 'selected box: ', selected_list_bboxes
    
    return selected_list_bboxes


###########################################################################
## AffordanceNet GET_MINIBATCH
def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
 
    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds) #im_blob la image sau khi scale ve 600 x X hoac X x 1000
 
    blobs = {'data': im_blob}
 
    if cfg.TRAIN.HAS_RPN: #=TRUE neu la end2end (xem file faster_rcnn_end2end.yml)
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0] #chuyen toa do cua box theo kich thuoc anh sau khi scale. [gt_inds, :] : vi 1 box la 1 array. boxes la 2D array
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds] #[gt_inds]: vi for each box, gt_classes la 1 scalar.
        blobs['gt_boxes'] = gt_boxes #matrix (1,5)
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],dtype=np.float32) #im_blob.shape[2]= height, im_blob.shape[3] = width
 
        ######################read mask index######################
        if cfg.TRAIN.MASK_REG:
            # print "====================Check roidb's fields======================"
            # print roidb[0]
            ''' --> in roidb[0] has: 
            {
                'gt_classes': array([ 7, 10,  1, 10,  7, 10], dtype=int32), 
                'max_classes': array([ 7, 10,  1, 10,  7, 10]), 
                'image': './data/coco/images/train2014/COCO_train2014_000000255576.jpg', 
                'bbox_targets': array([[  7.,   0.,   0.,   0.,   0.],
               [ 10.,   0.,   0.,   0.,   0.],
               [  1.,   0.,   0.,   0.,   0.],
               [ 10.,   0.,   0.,   0.,   0.],
               [  7.,   0.,   0.,   0.,   0.],
               [ 10.,   0.,   0.,   0.,   0.]], dtype=float32), 
               'seg_mask_inds': array([[255576,      1],
               [255576,      2],
               [255576,      3],
               [255576,      4],
               [255576,      5],
               [255576,      6]], dtype=uint32), 
               'flipped': False / True. Depend on this image is original or flipped
               'width': 640, ------> original image size
               'boxes': array([[114,  99, 541, 386],
               [253, 122, 263, 147],
               [472, 170, 518, 212],
               [142,  89, 167, 152],
               [ 39, 245,  68, 305],
               [266, 124, 280, 131]], dtype=uint16), 
               'max_overlaps': array([ 1.,  1.,  1.,  1.,  1.,  1.], dtype=float32), 
               'height': 427, ------> original image size
               'seg_areas': array([ 97571.140625  ,    230.68635559,   1110.08752441,   1616.10449219,
                 1478.43811035,    109.41735077], dtype=float32), 
                 'gt_overlaps': <6x81 sparse matrix of type '<type 'numpy.float32'>'
                with 6 stored elements in Compressed Sparse Row format>
            }
            '''
            seg_mask_inds = np.empty((len(gt_inds), 2), dtype=np.int32)
            seg_mask_inds[:, 0:2] = roidb[0]['seg_mask_inds'][gt_inds, :] #each gt_box has 1 correspondence seg_mask_ind
            blobs['seg_mask_inds'] = seg_mask_inds
 
            flipped = np.empty((1,1), dtype=np.bool_)
            flipped[0, 0] = roidb[0]['flipped']
            blobs['flipped'] = flipped
 
            # print "====================Check blobs's fields======================"
            # print blobs
            '''
            {
            'gt_boxes': array([[  64.78873444,  200.,  723.94366455,  487.32394409,   37.        ],
                                [ 100.        ,    1.40845072,  661.97180176,  307.04226685,    1.        ]], dtype=float32), 
            'data': array([[[[ 36.01990128,  46.75490189,  48.96990204, ..., -59.49517822,
              -45.75061035, -64.20959473],
             [ 41.66989899,  46.97807693,  50.60840225, ..., -60.74946594,
              -50.01625061, -64.46395111],
             [ 44.09490204,  46.77865601,  51.63615417, ..., -64.77590179,
              -55.13316727, -62.00203705],
             ..., 
             [ 44.99496841,  55.27793884,  66.12365723, ...,  -0.76256037,
               -9.7339344 ,   1.87395406],
             [ 41.4949379 ,  47.3398819 ,  53.23834991, ...,   3.3928864 ,
               -1.27210546,  10.27186584],
             [ 33.01990128,  34.71490097,  36.01990128, ...,   2.01990008,
                4.05017328,  13.98962593]],]]], dtype=float32), 
            'seg_mask_inds': array([[555074,      1],   [555074,      2]], dtype=int32), 
            'im_info': array([[ 600.        ,  901.        ,    1.40845072]], dtype=float32), #height/width of image after scaling.  1.40845072: used scale 
            'flipped': array([[False]], dtype=bool) # flipped = False --> image nay la original
           }
            '''
        ###########################################################
 
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)
 
            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))
 
            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))
 
        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)
 
        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob
 
        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)
 
    return blobs
########################################################################################



def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights


#########################################################
# Toan
def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image']) #read image
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]] #cfg.TRAIN.SCALES = 600 = target_size
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE) #cfg.TRAIN.MAX_SIZE = 1000
        im_scales.append(im_scale)
        processed_ims.append(im)
 
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
 
    return blob, im_scales
#########################################################



# #########################################################
# # Anh
# def _get_image_blob(roidb):
#     """Builds an input blob from the images in the roidb 
#     Return: blob image, list of crop_offsets  (since batch size = 1 --> len(list..) = 1
#     """
#     num_images = len(roidb)
#     #print 'LEN num_images: ', num_images # must be 1
#     processed_ims = []
#     #im_scales = []
#     crop_offsets = [] # keep list of crop offset (start_h, start_w) 
#     for i in xrange(num_images):
#         im = cv2.imread(roidb[i]['image']) #read image
#         if roidb[i]['flipped']:  ## TODO DISSABLE
#             im = im[:, ::-1, :]
#             
#         im, start_offsets = crop_pad_im_for_blob(im, cfg.PIXEL_MEANS, cfg.IMG_CROPPAD_SIZE)
#         
#         # add to list
#         crop_offsets.append(start_offsets)
#         processed_ims.append(im)
# 
#     # Create a blob to hold the input images
#     blob = im_list_to_blob(processed_ims)
# 
#     if print_verbose: print 'GOT ALL OFFSET: ', crop_offsets
#     
#     return blob, crop_offsets
# #########################################################


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()

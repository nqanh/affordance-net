# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import random

random.seed(999) # to reproduce

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size): #target_size = 600, max_size = 1000
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min) #im_scale = 600/im_size_min --> resize minsize ve 600
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size: #neu kich thuoc cua maxsize sau khi scale > 1000
        im_scale = float(max_size) / float(im_size_max) # im_scale = 1000/im_size_max: tinh lai im_scale de max size la 1000
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

# Anh
## crop and pad image randomly based on crop_size 
def crop_pad_im_for_blob(im, pixel_means, crop_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    #im_shape = im.shape
    
    height, width, _ = im.shape
    
    pad_height = max(crop_size - height, 0) ## do not use np.max(-3, 0) --> -3
    pad_width  = max(crop_size - width, 0)
    
#     print 'pad height: ', pad_height
#     print 'pad width : ', pad_width
    
    # pad image to crop_size, do nothing if pad=0 --> img still at original size
    im = cv2.copyMakeBorder(im, 0, pad_height,
                            0, pad_width, cv2.BORDER_CONSTANT,
                            value=[104.008, 116.669, 122.675])
    
#     print 'img after pad + pad shape: ', im.shape
    curr_h, curr_w, _ = im.shape # get current shape
    extra_h = curr_h - crop_size
    extra_w = curr_w - crop_size
    
    start_h, start_w = 0, 0 # crop index is 0 by default
    
#     if extra_h > 0:
#         start_h = random.choice(np.arange(extra_h))
#     if extra_w > 0:
#         start_w = random.choice(np.arange(extra_w))
    
    #print '------ extra_h: ', extra_h
    #print '------ extra_w: ', extra_w
    
    if extra_h > 3: # not > 0 to make sure random.arange get good number
        #start_h = random.choice(np.arange(extra_h)) # get from 0 to extra_h
        start_h = random.choice(np.arange(int(extra_h/3), int(extra_h/3) * 2))   ## MORE CENTER CROP!!! --> GET GOOD BBOXES!!!
    if extra_w > 3:
        #start_w = random.choice(np.arange(extra_w))
        start_w = random.choice(np.arange(int(extra_w/3), int(extra_w/3) * 2))
            
            
#     print 'start h: ', start_h
#     print 'start w: ', start_w
    
    croped_im = im[start_h:start_h+crop_size, start_w:start_w+crop_size]
#     print 'croped im size: ', croped_im.shape
    
#         cv2.imshow('crop im', croped_im)
#         cv2.waitKey(0)
#         cv2.imshow('original img: ', im)
#         cv2.waitKey(0)

    start_offsets = [start_h, start_w]
    
    return croped_im, start_offsets
    
    
    
    
    
    
    
    
    

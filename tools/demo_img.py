"""
See README.md for installation instructions before running.
Demo script to perform affordace detection from images
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import numpy as np
import os, cv2
import argparse

import caffe


CONF_THRESHOLD = 0.9
good_range = 0.005
    
# get current dir
cwd = os.getcwd()
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'AffordanceNet root folder: ', root_path
img_folder = cwd + '/img'

OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife', 'cup', 'drill', 'racket', 'spatula', 'bottle')

# Mask
background = [200, 222, 250]  
c1 = [0,0,205]   
c2 = [34,139,34] 
c3 = [192,192,128]   
c4 = [165,42,42]    
c5 = [128,64,128]   
c6 = [204,102,0]  
c7 = [184,134,11] 
c8 = [0,153,153]
c9 = [0,134,141]
c10 = [184,0,141] 
c11 = [184,134,0] 
c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

# Object
col0 = [0, 0, 0]
col1 = [0, 255, 255]
col2 = [255, 0, 255]
col3 = [0, 125, 255]
col4 = [55, 125, 0]
col5 = [255, 50, 75]
col6 = [100, 100, 50]
col7 = [25, 234, 54]
col8 = [156, 65, 15]
col9 = [215, 25, 155]
col10 = [25, 25, 155]

col_map = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]






def reset_mask_ids(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later 
    counter = 0
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
        
    return mask
    

    
def convert_mask_to_original_ids_manual(mask, original_uni_ids):
    #TODO: speed up!!!
    temp_mask = np.copy(mask) # create temp mask to do np.around()
    temp_mask = np.around(temp_mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
    current_uni_ids = np.unique(temp_mask)
     
    out_mask = np.full(mask.shape, 0, 'float32')
     
    mh, mw = mask.shape
    for i in range(mh-1):
        for j in range(mw-1):
            for k in range(1, len(current_uni_ids)):
                if mask[i][j] > (current_uni_ids[k] - good_range) and mask[i][j] < (current_uni_ids[k] + good_range):  
                    out_mask[i][j] = original_uni_ids[k] 
                    #mask[i][j] = current_uni_ids[k]
           
#     const = 0.005
#     out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
              
    #return mask
    return out_mask
        



def draw_arrow(image, p, q, color, arrow_magnitude, thickness, line_type, shift):
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    
def draw_reg_text(img, obj_info):
    #print 'tbd'
    
    obj_id = obj_info[0]
    cfd = obj_info[1]
    xmin = obj_info[2]
    ymin = obj_info[3]
    xmax = obj_info[4]
    ymax = obj_info[5]
    
    draw_arrow(img, (xmin, ymin), (xmax, ymin), col_map[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymin), (xmax, ymax), col_map[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmax, ymax), (xmin, ymax), col_map[obj_id], 0, 5, 8, 0)
    draw_arrow(img, (xmin, ymax), (xmin, ymin), col_map[obj_id], 0, 5, 8, 0)
    
    # put text
    txt_obj = OBJ_CLASSES[obj_id] + ' ' + str(cfd)
    cv2.putText(img, txt_obj, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1) # draw with red
    #cv2.putText(img, txt_obj, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col_map[obj_id], 2)
    
#     # draw center
#     center_x = (xmax - xmin)/2 + xmin
#     center_y = (ymax - ymin)/2 + ymin
#     cv2.circle(img,(center_x, center_y), 3, (0, 255, 0), -1)
    
    return img



def visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh):

    if rois_final.shape[0] == 0:
        print 'No detected box at all!'
        return

    inds = np.where(rois_class_score[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print 'No detected box with probality > thresh = ', thresh, '-- Choossing highest confidence bounding box.'
        inds = [np.argmax(rois_class_score)]  
        max_conf = np.max(rois_class_score)
        if max_conf < 0.001: 
            return  ## confidence is < 0.001 -- no good box --> must return
            

    rois_final = rois_final[inds, :]
    rois_class_score = rois_class_score[inds,:]
    rois_class_ind = rois_class_ind[inds,:]
    

    # get mask
    masks = masks[inds, :, :, :]
    
    im_width = im.shape[1]
    im_height = im.shape[0]
    
    # transpose
    im = im[:, :, (2, 1, 0)]

    num_boxes = rois_final.shape[0]
    
    list_bboxes = []

    
    for i in xrange(0, num_boxes):
        
        curr_mask = np.full((im_height, im_width), 0.0, 'float') # convert to int later
            
        class_id = int(rois_class_ind[i,0])
    
        bbox = rois_final[i, 1:5]
        score = rois_class_score[i,0]
        
        if cfg.TEST.MASK_REG:

            x1 = int(round(bbox[0]))
            y1 = int(round(bbox[1]))
            x2 = int(round(bbox[2]))
            y2 = int(round(bbox[3]))

            x1 = np.min((im_width - 1, np.max((0, x1))))
            y1 = np.min((im_height - 1, np.max((0, y1))))
            x2 = np.min((im_width - 1, np.max((0, x2))))
            y2 = np.min((im_height - 1, np.max((0, y2))))
            
            cur_box = [class_id, score, x1, y1, x2, y2]
            list_bboxes.append(cur_box)
            
            h = y2 - y1
            w = x2 - x1
            
            
            mask = masks[i, :, :, :]
            mask = np.argmax(mask, axis=0)
            
            
            original_uni_ids = np.unique(mask)
            
            # sort before_uni_ids and reset [0, 1, 7] to [0, 1, 2]
            original_uni_ids.sort()
            mask = reset_mask_ids(mask, original_uni_ids)
            
            mask = cv2.resize(mask.astype('float'), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            #mask = convert_mask_to_original_ids(mask, original_uni_ids)
            mask = convert_mask_to_original_ids_manual(mask, original_uni_ids)
            
            #FOR MULTI CLASS MASK
            curr_mask[y1:y2, x1:x2] = mask # assign to output mask
            
            # visualize each mask
            curr_mask = curr_mask.astype('uint8')
            color_curr_mask = label_colours.take(curr_mask, axis=0).astype('uint8')
            cv2.imshow('Mask' + str(i), color_curr_mask)
            #cv2.imwrite('mask'+str(i)+'.jpg', color_curr_mask)


    ori_file_path = img_folder + '/' + im_name 
    img_org = cv2.imread(ori_file_path)
    for ab in list_bboxes:
        print 'box: ', ab
        img_out = draw_reg_text(img_org, ab)
    
    cv2.imshow('Obj detection', img_out)
    #cv2.imwrite('obj_detction.jpg', img_out)
    cv2.waitKey(0)
    


def run_affordance_net(net, image_name):

    im_file = img_folder + '/' + im_name
    im = cv2.imread(im_file)
    
    ori_height, ori_width, _ = im.shape
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if cfg.TEST.MASK_REG:
        rois_final, rois_class_score, rois_class_ind, masks, scores, boxes = im_detect2(net, im)
    else:
        1
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, rois_final.shape[0])
    
    # Visualize detections for each class
    visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh=CONF_THRESHOLD)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='AffordanceNet demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    
    prototxt = root_path + '/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = root_path + '/pretrained/AffordanceNet_200K.caffemodel'   
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    # load network
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)


    list_test_img = os.walk(img_folder).next()[2]
    
    
    # run detection for each image
    for idx, im_name in enumerate(list_test_img):
        print '##########################################################'
        im_name = im_name.strip()
        #im_name = '5'
        print 'Current idx: ', idx, ' / ', len(list_test_img)
        print 'Current img: ', im_name
        run_affordance_net(net, im_name)


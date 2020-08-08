from __future__ import print_function

"""
Use the confidence of the object detection as the confidence for object classification.
If the predicted class is cup, pan, bowl, then the confidence is used as the confidence
of container.
If the predicted class is other classes, then the confidence is 0.
If there is no prediction, then the confidence is 0.

This code generates map for a single view among all the view of the 3D scanning.

Author: Hongtao Wu
July 5, 2020
"""

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
import json

import caffe


CONF_THRESHOLD = 0.9
good_range = 0.005
    
# get current dir
cwd = os.getcwd()
root_path = '/home/hongtao/src/affordance-net'  # get parent path
print ('AffordanceNet root folder: ', root_path)
# img_folder = cwd + '/img'
data_dir = '/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_benchmark_object'
class_folders = ['bowl', 'cup', 'drill', 'hammer', 'knife', 'pan', 'spatula']
# class_folders = ['cup']



OBJ_CLASSES = ('__background__', 'bowl', 'tvm', 'pan', 'hammer', 'knife', 'cup', 'drill', 'racket', 'spatula', 'bottle')

# Mask
background = [200, 222, 250]  
c1 = [0,0,205] # Contain
c2 = [34,139,34] # Cut
c3 = [192,192,128] # Display
c4 = [165,42,42] # Engine
c5 = [128,64,128] # grasp
c6 = [204,102,0] # hit
c7 = [184,134,11] # pound
c8 = [0,153,153] # support
c9 = [0,134,141] # w-grasp
c10 = [184,0,141] 
c11 = [184,134,0] 
c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

# Object (Bounding box)
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

    inds = np.where(rois_class_score[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print ('No detected box with probality > thresh = ', thresh, '-- Choossing highest confidence bounding box.')
        inds = [np.argmax(rois_class_score)]  
        max_conf = np.max(rois_class_score)
        if max_conf < 0.001: 
            return None, None, []  ## confidence is < 0.001 -- no good box --> must return
            

    rois_final = rois_final[inds, :]
    rois_class_score = rois_class_score[inds,:]
    rois_class_ind = rois_class_ind[inds,:]
    

    # get mask
    masks = masks[inds, :, :, :]
    
    im_width = im.shape[1]
    im_height = im.shape[0]
    
    im_ori = np.copy(im)
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
            # print "original_uni_ids: ", original_uni_ids
            stop = np.sum(original_uni_ids > 9)


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
            if stop:
                cv2.imshow('Mask' + str(i), color_curr_mask)
                cv2.imshow('Obj detection', img_out)
                cv2.waitKey(0)            
            # cv2.imwrite(os.path.join(benchmark_folder,'mask_' + str(i) + '_' + im_name), color_curr_mask)


    # ori_file_path = img_folder + '/' + im_name 
    # img_org = cv2.imread(ori_file_path)
    for ab in list_bboxes:
        print ('box: ', ab)
        img_out = draw_reg_text(im_ori, ab)
    
    
    # cv2.imshow('Obj detection', img_out)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(benchmark_folder, 'objdet_' + im_name), img_out)

    return color_curr_mask, img_out, list_bboxes
    


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
    
    # Visualize detections for each class
    color_cuur_mask, img_out, list_bboxes = visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh=CONF_THRESHOLD)

    return color_cuur_mask, img_out


def run_affordance_net_map(net, image_name):
    im_file = img_folder + '/' + im_name
    im = cv2.imread(im_file)

    ori_height, ori_width, _ = im.shape

    if cfg.TEST.MASK_REG:
        rois_final, rois_class_score, rois_class_ind, masks, scores, boxes = im_detect2(net, im)
    else:
        1

    # print "rois_final: ", rois_final
    # print "scores shape: ", scores.shape
    # print "masks shape: ", masks.shape
    # print "boxes shape: ", boxes.shape
    # print "rois_class_score: ", rois_class_score, rois_class_score.shape
    # print "rois_class_ind: ", rois_class_ind, rois_class_ind.shape
    
    assert rois_class_score.shape == rois_class_ind.shape

    # Use the bounding box with the largest score as the classification result.
    largest_cfd_idx = np.argmax(rois_class_score)
    # print "largest_cfd_idx: ", largest_cfd_idx

    obj_cfd = rois_class_score[largest_cfd_idx][0]
    obj_classification_idx = rois_class_ind[largest_cfd_idx][0]

    # Find the bounding box in scores
    largest_cfd_idx_in_score = np.where(scores==obj_cfd)

    # If there is no detection then the container cfd equals 0
    if rois_class_score[0][0] == -1:
        obj_iscontainer = False
        obj_container_cfd = 0.0
    else:
        # Open container idx: bowl(1), cup(6), pan(3)
        if obj_classification_idx == 1 or obj_classification_idx == 3 or obj_classification_idx == 6:
            # print "Object is classified as an open container"
            obj_iscontainer = True
        else:
            obj_iscontainer = False
        
        bbox_score_list = scores[largest_cfd_idx_in_score[0][0],:]
        # print "bbox_score_list sum: ", np.sum(bbox_score_list)
        bbox_container_score_list = np.array([bbox_score_list[1], bbox_score_list[3], bbox_score_list[6]])
        # print "bbox_container_score_list: ", bbox_container_score_list
        obj_container_cfd = np.max(bbox_container_score_list)

    # print "obj_iscontainer: ", obj_iscontainer
    # print "obj_container_cfd: ", obj_container_cfd

    color_curr_mask, img_out = visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh=CONF_THRESHOLD)

    return obj_iscontainer, obj_container_cfd, color_curr_mask, img_out


def run_affordance_net_map_direct_crop(net, crop_img):

    ori_height, ori_width, _ = crop_img.shape

    if cfg.TEST.MASK_REG:
        rois_final, rois_class_score, rois_class_ind, masks, scores, boxes = im_detect2(net, crop_img)
    else:
        1

    # print "rois_final: ", rois_final
    # print "scores shape: ", scores.shape
    # print "masks shape: ", masks.shape
    # print "boxes shape: ", boxes.shape
    # print ("rois_class_score: ", rois_class_score, rois_class_score.shape)
    # print ("rois_class_ind: ", rois_class_ind, rois_class_ind.shape)
    
    assert rois_class_score.shape == rois_class_ind.shape

    # Use the bounding box with the largest score as the classification result.
    largest_cfd_idx = np.argmax(rois_class_score)
    # print "largest_cfd_idx: ", largest_cfd_idx

    obj_cfd = rois_class_score[largest_cfd_idx][0]
    obj_classification_idx = rois_class_ind[largest_cfd_idx][0]

    # If there is no detection then the container cfd equals 0
    if rois_class_score[0][0] == -1:
        obj_iscontainer = False
        obj_container_cfd = 0.0
        # Nothing is detected
        bbox_score_list = []
    else:
        # Find the bounding box in scores
        largest_cfd_idx_in_score = np.where(scores==obj_cfd)
        print ("largest_cfd_idx_in_score: ", largest_cfd_idx_in_score)
        # Make sure there is only one such box
        assert largest_cfd_idx_in_score[0].shape[0] == 1
        assert largest_cfd_idx_in_score[1].shape[0] == 1
        # Open container idx: bowl(1), cup(6), pan(3)
        if obj_classification_idx == 1 or obj_classification_idx == 3 or obj_classification_idx == 6:
            # print "Object is classified as an open container"
            obj_iscontainer = True
        else:
            obj_iscontainer = False
        
        bbox_score_list = scores[largest_cfd_idx_in_score[0][0],:]
        # print "bbox_score_list sum: ", np.sum(bbox_score_list)
        bbox_container_score_list = np.array([bbox_score_list[1], bbox_score_list[3], bbox_score_list[6]])
        # print "bbox_container_score_list: ", bbox_container_score_list
        obj_container_cfd = np.max(bbox_container_score_list)

    color_curr_mask, img_out, list_bboxes = visualize_mask(crop_img, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh=CONF_THRESHOLD)

    # print "obj_iscontainer: ", obj_iscontainer
    # print "obj_container_cfd: ", obj_container_cfd

    return obj_iscontainer, obj_container_cfd, color_curr_mask, img_out, list_bboxes, bbox_score_list, rois_class_score, rois_class_ind

        

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
    caffemodel = os.path.join(root_path, 'pretrained', 'AffordanceNet_200K.caffemodel')   
    
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
    print ('\n\nLoaded network {:s}'.format(caffemodel))

    map_dir = "/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_map_0726"
    det_result_dir = "/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_map_result_0726"

    # Class
    for class_folder in class_folders:
        class_dir = os.path.join(data_dir, class_folder)
        
        object_folders = os.listdir(class_dir)
        # object_folders = ["Origami_Pink_Cup"]

        benchmark_objdet_class_dir = os.path.join(det_result_dir, class_folder)
        if os.path.exists(benchmark_objdet_class_dir):
            pass
        else:
            os.mkdir(benchmark_objdet_class_dir)

        # Object
        for object_folder in object_folders:
            obj_dir = os.path.join(class_dir, object_folder)
            img_folder = obj_dir # the code needs this parameter to find the image            
            img_files = os.listdir(obj_dir)
            
            benchmark_objdet_obj_dir = os.path.join(benchmark_objdet_class_dir, object_folder)
            if os.path.exists(benchmark_objdet_obj_dir):
                pass
            else:
                os.mkdir(benchmark_objdet_obj_dir)

            bbox_json = object_folder + "_bbox.json"
            json_path = os.path.join(obj_dir, bbox_json)
            rgbd_dir = os.path.join(obj_dir, 'rgbd')
            with open(json_path) as f:
                bbox_dict = json.load(f)
                img_num = 0
                total_img_num = 24
                for (key, value) in bbox_dict.items():
                    img_num += 1
                    img_name = value["filename"]
                    img = cv2.imread(os.path.join(rgbd_dir, img_name))

                    print (object_folder)
                    print (img_name)

                    img_idx = img_name.split('.')[0]

                    frame_map_dir = os.path.join(map_dir, img_idx)
                    if not os.path.exists(frame_map_dir):
                        os.mkdir(frame_map_dir)
                    
                    img_h = img.shape[0]
                    img_w = img.shape[1]
                    print ("img_h, img_w: ", img_h, img_w)

                    x = value["regions"][0]["shape_attributes"]["x"]
                    y = value["regions"][0]["shape_attributes"]["y"]
                    width = value["regions"][0]["shape_attributes"]["width"]
                    height = value["regions"][0]["shape_attributes"]["height"]

                    # print "x, y: ", x, y
                    # print "width, height: ", width, height

                    x_1 = max(0, x)
                    y_1 = max(0, y)

                    x_2 = min(x+width, img_w)
                    y_2 = min(y+height, img_h)

                    print ("x1, x2, y1, y2: {}, {}, {}, {}".format(x_1, x_2, y_1, y_2))
                    crop_img = img[y_1:y_2, x_1:x_2]
                    crop_img_name = img_name.split(".")[0] + ".crop.png"
                    crop_img_path = os.path.join(benchmark_objdet_obj_dir, crop_img_name)
                    cv2.imwrite(crop_img_path, crop_img)

                    im_name = img_name
                    obj_iscontainer, obj_container_cfd, color_curr_mask, img_out, list_bboxes, bbox_score_list, rois_class_score, rois_class_ind = run_affordance_net_map_direct_crop(net, crop_img)
                    
                    # print "obj_iscontainer: ", obj_iscontainer
                    # print "obj_container_cfd: ", obj_container_cfd
                    
                    obj_name = object_folder
                    map_filename = obj_name + ".txt"
                    classification_filename = obj_name + "_classification.txt"
                    map_path = os.path.join(frame_map_dir, map_filename)
                    classification_path = os.path.join(frame_map_dir, classification_filename)
                    with open(map_path, 'w') as f2:
                        writerow = "container " + str(obj_container_cfd) + " 0 1 2 3"                   
                        f2.write(writerow)
                    with open(classification_path, 'w') as f3:
                        if obj_iscontainer:
                            writerow = "container"
                        else:
                            writerow = "noncontainer"
                        f3.write(writerow)

                    img_filename = img_name.split(".")[0]
                    mask_filename = img_filename + ".mask.png"
                    mask_path = os.path.join(benchmark_objdet_obj_dir, mask_filename)
                    cv2.imwrite(mask_path, color_curr_mask)
                    objdet_filename = img_filename + ".objdet.png"
                    objdet_path = os.path.join(benchmark_objdet_obj_dir, objdet_filename)
                    cv2.imwrite(objdet_path, img_out)

                    # txt_file = img_filename + "_running.txt"
                    # with open(os.path.join(benchmark_objdet_obj_dir, txt_file), 'a') as f1:
                    #     print ("object folders: ", object_folders, file=f1)
                    #     print ("rois_class_score: ", rois_class_score, file=f1)
                    #     print ("rois_class_ind: ", rois_class_ind, file=f1)
                    #     for ab in list_bboxes:
                    #         print("box: ", ab, file=f1)
                    #     print ("selected bbox score list: ", bbox_score_list, file=f1)
                    #     print ("object container cfd: ", obj_container_cfd, file=f1)

                    print ("======")
        
        assert img_num == total_img_num

            # for img_file in img_files:
            #     print 'Current img: ', os.path.join(obj_dir, img_file)
            #     img_idx = img_file.split('.')[0]

            #     frame_map_dir = os.path.join(map_dir, img_idx)
            #     if not os.path.exists(frame_map_dir):
            #         os.mkdir(frame_map_dir)

            #     im_name = img_file
            #     obj_name = object_folder
            #     obj_iscontainer, obj_container_cfd, color_curr_mask, img_out = run_affordance_net_map(net, im_name, obj_name)
            #     print "======"
            #     map_filename = obj_name + ".txt"
            #     map_path = os.path.join(frame_map_dir, map_filename)
            #     with open(map_path, 'w') as f:
            #         writerow = "container " + str(obj_container_cfd) + " 0 1 2 3"                   
            #         f.write(writerow)

            #     img_filename = img_file.split(".")[0]
            #     mask_filename = img_filename + ".mask.png"
            #     mask_path = os.path.join(benchmark_objdet_obj_dir, mask_filename)
            #     cv2.imwrite(mask_path, color_curr_mask)
            #     objdet_filename = img_filename + ".objdet.png"
            #     objdet_path = os.path.join(benchmark_objdet_obj_dir, objdet_filename)
            #     cv2.imwrite(objdet_path, img_out)
                
                    

                
                
                



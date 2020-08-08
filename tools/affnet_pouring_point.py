"""
See README.md for installation instructions before running.
Demo script to perform affordace detection from images
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import json
import numpy as np
import os, cv2
import argparse
import caffe



CONF_THRESHOLD = 0.9
good_range = 0.005
    
# get current dir
cwd = os.getcwd()
root_path = "/home/hongtao/src/affordance-net" # get parent path
print 'AffordanceNet root folder: ', root_path
# img_folder = cwd + '/img'

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

    # cv2.imshow("draw_reg_text.png", img)
    # cv2.waitKey(0)
    
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



def visualize_mask_asus(object_name, im, rois_final, rois_class_score, rois_class_ind, masks, thresh):

    list_bboxes = []
    list_masks = []

    if rois_final.shape[0] == 0:
        print 'No object detection!'
        return list_bboxes, list_masks
    
    inds = np.where(rois_class_score[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print 'No detected box with probality > thresh = ', thresh, '-- Choossing highest confidence bounding box.'
        inds = [np.argmax(rois_class_score)]  
        max_conf = np.max(rois_class_score)
        if max_conf < 0.001: 
            return list_bboxes, list_masks
            
    rois_final = rois_final[inds, :]
    rois_class_score = rois_class_score[inds,:]
    rois_class_ind = rois_class_ind[inds,:]
    
    # get mask
    masks = masks[inds, :, :, :]
    
    im_width = im.shape[1]
    im_height = im.shape[0]
    
    # Copy img before it is transposed
    img_org = im.copy()

    # transpose
    im = im[:, :, (2, 1, 0)]

    num_boxes = rois_final.shape[0]
    
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
            
            # for mult masks
            curr_mask[y1:y2, x1:x2] = mask 
            
            # visualize each mask
            curr_mask = curr_mask.astype('uint8')
            list_masks.append(curr_mask)
            color_curr_mask = label_colours.take(curr_mask, axis=0).astype('uint8')
            # cv2.imshow('Mask' + str(i), color_curr_mask)
            mask_filename = object_name + '_aff_mask' + str(i) + '.jpg'
            cv2.imwrite(os.path.join(img_folder, mask_filename), color_curr_mask)

    for ab in list_bboxes:
        print 'box: ', ab
        img_out = draw_reg_text(img_org, ab)
    
    # cv2.imshow('Obj Detection', img_out)
    obj_det_filename = object_name + '_aff_det.png'
    cv2.imwrite(os.path.join(img_folder, obj_det_filename), img_out)
    #cv2.waitKey(0)
    
    return list_bboxes, list_masks


def get_list_centroid(current_mask, obj_id):
    list_uni_ids = list(np.unique(current_mask))
    list_uni_ids.remove(0) ## remove background id
    
    list_centroid = []  ## each row is: obj_id, mask_id, xmean, ymean
    for val in list_uni_ids:
        inds = np.where(current_mask == val) 
        x_index = inds[1]
        y_index = inds[0]
        
        xmean = int(np.mean(x_index))
        ymean = int(np.mean(y_index))
        
        cur_centroid = [obj_id, val, xmean, ymean]
        list_centroid.append(cur_centroid)
        
    return list_centroid   

def convert_bbox_to_centroid(list_boxes, list_masks):
    assert len(list_boxes) == len(list_masks), 'ERROR: len(list_boxes) and len(list_masks) must be equal'
    list_final = []
    for i in range(len(list_boxes)):
        obj_id = list_boxes[i][0] 
        list_centroids = get_list_centroid(list_masks[i], obj_id)  # return [[obj_id, mask_id, xmean, ymean]]
        if len(list_centroids) > 0:
            for l in list_centroids:
                list_final.append(l)
        
    return list_final


def select_object_and_aff(list_obj_centroids, obj_id, aff_id):
    # select the first object with object id and aff id
    selected_obj_aff = []
    for l in list_obj_centroids:
        if len(l) > 0:
            if l[0] == obj_id and l[1] == aff_id:
                selected_obj_aff.append(l)
                break
    
    selected_obj_aff = np.squeeze(selected_obj_aff, 0)
    return selected_obj_aff  

    
def project_to_3D(width_x, height_y, depth, ic):
    X = (width_x - ic.asus_cx) * dval / ic.asus_fx      
    Y = (height_y - ic.asus_cy) * dval / ic.asus_fy
    Z = depth
    p3D = [X, Y, Z]
    
    return p3D


def project_to_3D_pour(width_x, height_y, depth, cx, cy, fx, fy):
    """
    cx, cy, fx, fy are all in the unit of pixel
    depth is in the unit of mm
    """
    X = (width_x - cx) * dval / fx      
    Y = (height_y - cy) * dval / fy
    Z = depth
    p3D = [X, Y, Z]
    
    return p3D


def run_affordance_net_asus(object_name, net, im):
    # cv2.imshow("run_affordance_net_asus.png",im)
    # cv2.waitKey(0)
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
    return visualize_mask_asus(object_name, im, rois_final, rois_class_score, rois_class_ind, masks, thresh=CONF_THRESHOLD)


def read_pose_txt(pose_txt):
    """
    Read the pose txt file and return a 4x4 rigid transformation.
    """
    with open(pose_txt, "r") as f:
        lines = f.readlines()
        pose = np.zeros((4, 4))
        for line_idx, line in enumerate(lines):
            items = line.split(" ")
            for i in range(4):
                pose[line_idx, i] = float(items[i])
    
    return pose




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

    contain_affordance_idx = 1
    PS_fx = 5.32580000e+02
    PS_fy = 5.32210000e+02
    PS_cx = 3.16040000e+02
    PS_cy = 2.36030000e+02
    
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

    
    data_folder = "/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_benchmark_pouring_3DScanning"
    object_name = "Ikea_Framkalla_Mug_Pink_3D"
    img_folder = os.path.join(data_folder, object_name)
    full_rgb = object_name + "_affnet_pour_rgb.png"
    full_dep = object_name + "_affnet_pour_depth.png"
    cam_pose_txt = object_name + "_affnet_pour_pose.txt"
    bbox_json = object_name + "_affnet_pour_bbox.json"
    pour_point_txt = object_name + "_affnet_pour_point.txt"

    with open(os.path.join(img_folder, bbox_json)) as f:
        bbox_dict = json.load(f)
        img_num = 0
        for (key, value) in bbox_dict.items():
            img_num += 1
            img_name = value["filename"]
            full_rgb_img = cv2.imread(os.path.join(img_folder, full_rgb))
            full_depth_img = cv2.imread(os.path.join(img_folder, full_dep), cv2.IMREAD_UNCHANGED)

            print "rgb, depth: ", full_rgb, ", ", full_dep
            
            img_h = full_rgb_img.shape[0]
            img_w = full_rgb_img.shape[1]
            print "img_h, img_w: ", img_h, img_w

            x = value["regions"][0]["shape_attributes"]["x"]
            y = value["regions"][0]["shape_attributes"]["y"]
            width = value["regions"][0]["shape_attributes"]["width"]
            height = value["regions"][0]["shape_attributes"]["height"]

            print "x, y: ", x, y
            print "width, height: ", width, height
            print "======="

            x_1 = max(0, x)
            y_1 = max(0, y)

            x_2 = min(x+width, img_w)
            y_2 = min(y+height, img_h)
            
            rgb = full_rgb_img[y_1:y_2, x_1:x_2]
            dep = full_depth_img[y_1:y_2, x_1:x_2]

            crop_rgb_filename = object_name + "_affnet_pour_crop_rgb.png"
            crop_depth_filename = object_name + "_affnet_pour_crop_depth.png"

            cv2.imwrite(os.path.join(img_folder, crop_rgb_filename), rgb)
            cv2.imwrite(os.path.join(img_folder, crop_depth_filename), dep)
        
        assert img_num == 1
        width_offset = x_1
        height_offset = y_1


    # if (rgb != None and dep != None):
    #     print 'rgb shape: ', rgb.shape
    #     h, w, c = rgb.shape
        
    #     if (h > 100 and w > 100):
    #         print '--------------------------------------------------------'
    #         # cv2.imshow('Input Image', rgb)     
    #         # cv2.waitKey(0)
            
    # run detection
    list_boxes, list_masks = run_affordance_net_asus(object_name, net, rgb)
    print "list_boxes: ", list_boxes

    print 'len list boxes: ', len(list_boxes)
    if len(list_boxes) < 1:
        # no object found, count as fail in pouring
        print "No object is found!" 
    else:
        if len(list_boxes) > 1:
            # There are more than 1 bbox
            # Choose the bbox with the highest scores
            np_list_boxes = np.array(list_boxes)
            box_scores = np_list_boxes[:, 1]
            largest_score_idx = np.argmax(box_scores)
            list_boxes = [list_boxes[largest_score_idx]]
            list_masks = [list_masks[largest_score_idx]]
        
        print "filtered list boxes: ", list_boxes
        print "filtered list masks length: ", len(list_masks)
        
        list_obj_centroids = convert_bbox_to_centroid(list_boxes, list_masks)

        print "list_obj_centroids: ", list_obj_centroids

        np_obj_centroids = np.array(list_obj_centroids)
        affordance_list = np_obj_centroids[:, 1]
        
        if np.sum(affordance_list==contain_affordance_idx) >= 1:
            # has contain affordance
            contain_obj_centroid_idx = np.where(affordance_list==contain_affordance_idx)
            print "contain_obj_centroid_idx: ", contain_obj_centroid_idx

            pour_centroid = np_obj_centroids[contain_obj_centroid_idx[0][0]]
            print "pour_centroid: ", pour_centroid

            dval = dep[pour_centroid[3], pour_centroid[2]]
            width_x_origin = pour_centroid[2] + width_offset
            height_y_origin = pour_centroid[3] + height_offset

            cv2.circle(full_rgb_img, (width_x_origin, height_y_origin), 3, (255, 0, 0), 2)
            pour_point_filename = object_name + "_affnet_pour_point.png"
            cv2.imwrite(os.path.join(img_folder, pour_point_filename), rgb)

            p3D = project_to_3D_pour(width_x_origin, height_y_origin, dval, PS_cx, PS_cy, PS_fx, PS_fy) # unit: mm
            p3D = np.array(p3D) / 1000 # unit: m
            print "points in 3D: ", p3D

            cam_pose = read_pose_txt(os.path.join(img_folder, cam_pose_txt))
            print "cam_pose: ", cam_pose

            p3D = np.append(p3D, [1])
            p3D = p3D.reshape((4, 1))
            p3D_world = np.dot(cam_pose, p3D).flatten()

            print "p3D in world: ", p3D_world
            # Save the pouring point in the txt
            with open(os.path.join(img_folder, pour_point_txt), "w") as f:
                writerow = str(p3D_world[0]) + " " + str(p3D_world[1]) + " " + str(p3D_world[2])
                f.write(writerow)                
                 

                 
                    
                
                # # select object and affordance to project to 3D
                # obj_id = 10 # bottle
                # aff_id = 5  # grasp affordance
            
                # selected_obj_aff = select_object_and_aff(list_obj_centroids, obj_id, aff_id) 
                # if len(selected_obj_aff) < 1:
                #     continue
                
                # # get depth value from depth map                        
                # dval = dep[selected_obj_aff[3], selected_obj_aff[2]]
                
                # if dval != 'nan':
                #     # find 3D point
                #     p3Dc = project_to_3D(selected_obj_aff[2], selected_obj_aff[3], dval, ic)
                #     obj_pose_3D = PoseStamped()
                #     obj_pose_3D.header.frame_id = "camera_depth_optical_frame"
                    
                #     obj_pose_3D.pose.position.x = p3Dc[0]
                #     obj_pose_3D.pose.position.y = p3Dc[1]
                #     obj_pose_3D.pose.position.z = p3Dc[2]
                #     obj_pose_3D.pose.orientation.x = 0
                #     obj_pose_3D.pose.orientation.y = 0
                #     obj_pose_3D.pose.orientation.z = 0
                #     obj_pose_3D.pose.orientation.w = 1 ## no rotation
                #     # publish pose
                #     pub_obj_pose_3D.publish(obj_pose_3D)
"""
Benchmarking each view of the 24 capturing view.
The mask result for each object should be rearranged with rearrange_images.py before benchmarking

Author: Hongtao Wu
June 23, 2020
"""
from __future__ import division

import cv2
import os
import numpy as np

affnet_benchmark_object_dir = "/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_benchmark_object"
affnet_benchmark_view_dir = "/home/hongtao/Dropbox/ICRA2021/affnet_benchmark/affnet_benchmark_view"
class_folders = ['bowl', 'cup', 'drill', 'hammer', 'knife', 'pan', 'spatula']
class_dirs = [os.path.join(affnet_benchmark_object_dir, object_folder) for object_folder in class_folders]

total_benchmark_object_num = 34
view_num = 24
base_frame_idx = 150
benchmark_folder = "affnet_benchmark_0623"
contain_mask = np.array([0, 0, 205])

# Single view
for i in range(14, 15):
    correct_num = 0
    curr_frame_idx = base_frame_idx + i
    curr_frame_folder = os.path.join(affnet_benchmark_view_dir, benchmark_folder, str(curr_frame_idx))
    curr_frame_folder_objects = os.listdir(curr_frame_folder)

    for curr_frame_folder_object  in curr_frame_folder_objects:
        img_files = os.listdir(os.path.join(curr_frame_folder, curr_frame_folder_object))
        tested_correct = False
        for img_file in img_files:
            if 'mask' in img_file:
                img_path = os.path.join(curr_frame_folder, curr_frame_folder_object, img_file)
                img = cv2.imread(img_path)

                gt_iscup = ('Cup' in curr_frame_folder_object) or ('Mug' in curr_frame_folder_object)
                gt_isbowl = 'Bowl' in curr_frame_folder_object
                gt_ispan = ('pan' in curr_frame_folder_object) or ('Pan' in curr_frame_folder_object)
                gt_iscontainer = gt_iscup or gt_isbowl or gt_ispan

                iscontainer =  np.any(np.all(img == contain_mask, axis=2))

                if iscontainer:
                    print curr_frame_folder_object , " is a container"
                
                if gt_iscontainer:
                    print curr_frame_folder_object, ' is a container gt'
                    print "====="

                if iscontainer == gt_iscontainer:
                    # print curr_frame_folder_object, 'is detected correctly'
                    correct_num += 1                    
                    tested_correct = True

            # If an object is tested correct already
            # do not have to tested twice for a specific object
            if tested_correct:
                break

    containerbility_accuracy = correct_num / total_benchmark_object_num
    print "View {}: {}".format(curr_frame_idx, containerbility_accuracy)


# All view
correct_num = 0
for class_dir in class_dirs:
    object_folders = os.listdir(class_dir)

    for object_folder in object_folders:
        tested_correct = False

        gt_iscup = ('Cup' in object_folder) or ('Mug' in object_folder)
        gt_isbowl = 'Bowl' in object_folder
        gt_ispan = ('pan' in object_folder) or ('Pan' in object_folder)
        gt_iscontainer = gt_iscup or gt_isbowl or gt_ispan

        img_dir = os.path.join(class_dir, object_folder, benchmark_folder)
        img_files = os.listdir(img_dir)

        iscontainer = None

        for img_file in img_files:
            if 'mask' in img_file:
                img_path = os.path.join(img_dir, img_file)
                img = cv2.imread(img_path)
                iscontainer_img = np.any(np.all(img == contain_mask, axis=2))
                
                if iscontainer is None:
                    iscontainer = iscontainer_img
                else:
                    iscontainer = iscontainer and iscontainer_img 

        if iscontainer == gt_iscontainer:
            correct_num += 1

containerbility_accuracy = correct_num / total_benchmark_object_num
print "All: {}".format(containerbility_accuracy)

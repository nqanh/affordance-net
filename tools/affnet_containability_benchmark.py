"""
Benchmarking each view of the 24 capturing view.
The mask result for each object should be rearranged with rearrange_images.py before benchmarking

Author: Hongtao Wu
June 23, 2020
"""
from __future__ import division

import cv2
import os

affnet_benchmark_dir = "/home/hongtao/Dropbox/ICRA2021/data/affnet_benchmark"
affnet_benchmark_view_dir = "/home/hongtao/Dropbox/ICRA2021/data/affnet_benchmark_view"
class_folders = ['bowl', 'cup', 'drill', 'hammer', 'knife', 'pan', 'spatula']
class_dirs = [os.path.join(affnet_benchmark_dir, object_folder) for object_folder in class_folders]

total_benchmark_object_num = 34
view_num = 24
base_frame_idx = 150
benchmark_folder = "affnet_benchmark_0623"
container_mask = [0, 0, 205]


for i in range(view_num):
    correct_num = 0
    curr_frame_idx = base_frame_idx + i
    curr_frame_folder = os.path.join(affnet_benchmark_view_dir, benchmark_folder, str(curr_frame_idx))
    curr_frame_folder_objects = os.listdir(curr_frame_folder)
    if len(curr_frame_folder_objects) < total_benchmark_object_num:
        pass
    else:
        for curr_frame_folder_object  in curr_frame_folder_objects:
            img_files = os.listdir(os.path.join(curr_frame_folder, curr_frame_folder_object))
            for img_file in img_files:
                if 'mask' in img_file:
                    img_path = os.path.join(curr_frame_folder, curr_frame_folder_object, img_file)
                    img = cv2.imread(os.path.join(curr_frame_folder, curr_frame_folder_object, img_file))
                    if container_mask in img:
                        if ('cup' in curr_frame_folder_object) or ('pan' in curr_frame_folder_object) or ('bowl' in curr_frame_folder_object):
                            correct_num += 1
                    else:
                        if ('cup' in curr_frame_folder_object) or ('pan' in curr_frame_folder_object) or ('bowl' in curr_frame_folder_object):
                            pass
                        else:
                            correct_num += 1

    containerbility_accuracy = correct_num / total_benchmark_object_num
    print "View {}: {}".format(curr_frame_idx, containerbility_accuracy)

                    # cv2.imshow('mask.png', img)
                    # cv2.waitKey(0)

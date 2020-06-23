"""
Rearrange the images generated from affnet_benchmark_imagination.py
In total, there are 24 views. Make a directory for each view.
Within each view, make a directory for each objects.
Move the mask frame(s) and object detection frames into the view/object directory

Author: Hongtao Wu
June 23, 2020
"""

import os
import shutil

affnet_benchmark_dir = "/home/hongtao/Dropbox/ICRA2021/data/affnet_benchmark"
affnet_benchmark_view_dir = "/home/hongtao/Dropbox/ICRA2021/data/affnet_benchmark_view"
class_folders = ['bowl', 'cup', 'drill', 'hammer', 'knife', 'pan', 'spatula']
class_dirs = [os.path.join(affnet_benchmark_dir, object_folder) for object_folder in class_folders]

view_num = 24
base_frame_idx = 150
benchmark_folder = "affnet_benchmark_0623"

# Make a folder for each view
for i in range(view_num):
    curr_frame_idx = base_frame_idx + i
    curr_frame_folder = os.path.join(affnet_benchmark_view_dir, benchmark_folder, str(curr_frame_idx))
    if os.path.exists(curr_frame_folder):
        continue
    else:
        os.mkdir(curr_frame_folder)

# Move corresponding files of each view to the corresponding view folder
for class_dir in class_dirs:
    object_folders = os.listdir(class_dir)
    for object_folder in object_folders:
        affnet_benchmark_img_dir = os.path.join(class_dir, object_folder, benchmark_folder)
        affnet_benchmark_img_dir_files = os.listdir(affnet_benchmark_img_dir)

        for affnet_benchmark_img_dir_file in affnet_benchmark_img_dir_files:
            file_idx = int(affnet_benchmark_img_dir_file.split('-')[-1].split('.')[0])
            copy_target_dir = os.path.join(affnet_benchmark_view_dir, benchmark_folder, str(file_idx), object_folder)
            if os.path.exists(copy_target_dir):
                pass
            else:
                os.mkdir(copy_target_dir)
            copy_source_file = os.path.join(affnet_benchmark_img_dir, affnet_benchmark_img_dir_file)
            print 'source: ', copy_source_file
            print 'target: ', copy_target_dir
            shutil.copy2(copy_source_file, copy_target_dir)

# Check the number of object for a particular view
for i in range(view_num):
    curr_frame_idx = base_frame_idx + i
    curr_frame_folder = os.path.join(affnet_benchmark_view_dir, benchmark_folder, str(curr_frame_idx))
    curr_frame_folder_objects = os.listdir(curr_frame_folder)
    print curr_frame_idx, ": ", len(curr_frame_folder_objects)

            
            
            


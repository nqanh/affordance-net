# convert each instance mask (.png) to .sm (pkl file) format for AffordanceNet

## Input: 1 multiclass affordance mask (.png) of an object. The input .png should have the same size with the original image. See instance_vis folder for an example visualization.
## Output: 1 multiclass affordance mask (.sm) in pkl format.

### Notes: The index of object starts from 0, BUT the index of affordance mask starts from 1.
### E.g.,: The image '0.png' has 3 objects --> has 3 affordance masks: '0_1.png', '0_2.png', '0_3.png'. 
###        This script will convert these masks to '0_1.sm', '0_2.sm', and '0_3.sm' for AffordanceNet.


import os
import cv2
import cPickle

# get current dir
cwd = os.getcwd()
in_png_folder = cwd + '/instance_png'
out_sm_folder = cwd + '/instance_sm'
if not os.path.exists(out_sm_folder):
    os.makedirs(out_sm_folder)


def main():
    list_png = os.walk(in_png_folder).next()[2]
    list_png.sort()
    
    for l in list_png:
        print '-----------------------------------------------------'
        print 'current file: ', l
        full_png_file = os.path.join(in_png_folder, l)
        print 'full path: ', full_png_file
        mask_img = cv2.imread(full_png_file)
        print 'mask img shape: ', mask_img.shape
        mask_val = mask_img[:,:,0]  # just get 1 channel (both 3 channels are the same)
        print 'mask val shape: ', mask_val.shape
        
        out_name = l.replace('.png', '_segmask.sm')
        out_sm_file = os.path.join(out_sm_folder, out_name)
        print 'out sm file: ', out_sm_file
        
        ## write each mask to sm file
        with open(out_sm_file, 'wb') as f_seg_save:
            cPickle.dump(mask_val, f_seg_save, cPickle.HIGHEST_PROTOCOL)
        
        ## debug
        #break

if __name__ == '__main__':
    main()
    
    print 'ALL DONE!'
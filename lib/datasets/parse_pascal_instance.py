import os
import sys
import numpy as np
import scipy as sp
import scipy.io as spio
import cPickle

from lxml import etree as ET

# background = [135,206, 250]  #Light Sky Blue
background = [200, 222, 250]  # Light Sky Blue
c1 = [0, 0, 205]  # ok
c2 = [34, 139, 34]  # ok
c3 = [192, 192, 128]  # 3
c4 = [165, 42, 42]  # ok
c5 = [128, 64, 128]  # 5
c6 = [204, 102, 0]  # 6
c7 = [184, 134, 11]  # ok
c8 = [0, 153, 153]  # ok
c9 = [0, 134, 141]  # ok
c10 = [184, 0, 141]  # ok
c11 = [184, 134, 0]  # ok
c12 = [184, 134, 223]  # ok
c13 = [43, 134, 141]  # ok
c14 = [11, 23, 141]  # ok
c15 = [14, 34, 141]  # ok
c16 = [14, 134, 41]  # ok
c17 = [233, 14, 241]  # ok
c18 = [182, 24, 241]  # ok
c19 = [123, 13, 141]  # ok
c20 = [13, 164, 141]  # ok
c21 = [84, 174, 141]  # ok
c22 = [184, 14, 41]  # ok
c23 = [184, 34, 231]  # ok

label_colours = np.array(
    [background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23])
# label_mono = np.array([0, 1, 2 , 3, 4, 5, 6 ,7, 8, 9])


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

## CHANGE HERE
#pascal_instance_folder = '/home/anguyen/workspace/dataset/PASCALSEG/benchmark_RELEASE/dataset'  # path to dataset (download from Drive)
#out_folder = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/PASCAL_INSTANCE_FCIS'  # any output path
pascal_instance_folder = '/home/tdo/Software/PASCALSEG/benchmark_RELEASE/dataset'  # path to dataset (download from Drive)
out_folder = '/home/tdo/Software/VOCdevkit/VOC2012_FCIS'  # any output path

in_instance_folder = os.path.join(pascal_instance_folder, 'inst')


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


out_train_xml_path = os.path.join(out_folder, 'train', 'Annotations')
out_train_instance_pkl_path = os.path.join(out_folder, 'train', 'instance_pkl')
out_train_instance_vis_path = os.path.join(out_folder, 'train', 'instance_vis')

out_val_xml_path = os.path.join(out_folder, 'val', 'Annotations')
out_val_instance_pkl_path = os.path.join(out_folder, 'val', 'instance_pkl')
out_val_instance_vis_path = os.path.join(out_folder, 'val', 'instance_vis')

create_folder(out_train_xml_path)
create_folder(out_train_instance_pkl_path)
create_folder(out_train_instance_vis_path)

create_folder(out_val_xml_path)
create_folder(out_val_instance_pkl_path)
create_folder(out_val_instance_vis_path)


def writeXML(xmlpath, im_name, im_h, im_w, im_c, allBBoxes):
    # main structure
    anno = ET.Element('annotation')
    folder = ET.SubElement(anno, 'folder')
    fname = ET.SubElement(anno, 'filename')
    size = ET.SubElement(anno, 'size')

    # for folder
    folder.text = 'VOC2012'

    # for file name
    fname.text = im_name

    # for size block
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    height.text = str(im_h)
    width.text = str(im_w)
    depth.text = str(im_c)

    # for object block --> write bouding box
    for line in allBBoxes:
        # create an 'object' note
        obj = ET.SubElement(anno, 'object')
        oname = ET.SubElement(obj, 'name')
        opose = ET.SubElement(obj, 'pose')
        otrunc = ET.SubElement(obj, 'truncated')
        diffi = ET.SubElement(obj, 'difficult')
        obbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(obbox, 'xmin')
        ymin = ET.SubElement(obbox, 'ymin')
        xmax = ET.SubElement(obbox, 'xmax')
        ymax = ET.SubElement(obbox, 'ymax')

        # parse line to get info
        # ldata = line.split(' ')

        cur_box = line
        obj_id = cur_box[0]
        print 'OBJECT ID: ', obj_id

        # add to xml text
        # oname.text = str(ldata[0])  # write as ID --> DO NOT USE
        oname.text = classes[cur_box[0] - 1]  # write as name instead of ID
        opose.text = 'Unspecified'
        otrunc.text = '0'
        diffi.text = '0'  # set dfficult to 0 for all object

        #         # increase index of BB by 1 to as Pascal format
        #         xmin.text = str(int(float(ldata[1])) + 1)
        #         ymin.text = str(int(float(ldata[2])) + 1)
        #         xmax.text = str(int(float(ldata[3])) + 1)
        #         ymax.text = str(int(float(ldata[4])) + 1)

        # NO INCREASE
        xmin.text = str(int(cur_box[1]))
        ymin.text = str(int(cur_box[2]))
        xmax.text = str(int(cur_box[3]))
        ymax.text = str(int(cur_box[4]))

    # write to screen
    # print ET.tostring(anno, pretty_print=True, xml_declaration=False)
    # write to file:
    tree = ET.ElementTree(anno)
    tree.write(xmlpath, pretty_print=True, xml_declaration=False)


def parse_instance(list_in, out_xml_path, out_instance_pkl_path, out_instance_vis_path):
    for fm in list_in:
        # fm = '2008_000115.mat'

        # file_name = fm[0:len(fm)-4]
        file_name = fm
        print 'file name: ', file_name

        in_mat_file = os.path.join(in_instance_folder, file_name + '.mat')
        mat_data = spio.loadmat(in_mat_file)
        # print 'mat data: ', mat_data

        seg_img = mat_data['GTinst']['Segmentation']
        # print 'seg img dta: ', seg_img

        boundaries = mat_data['GTinst']['Boundaries']
        # print 'boundaries 1: ', boundaries
        #     boundaries2 = boundaries[0][0][0][0]
        #     boundaries2 = boundaries[0][0][1][0]
        #     print 'boundaries 2: ', boundaries2
        #     print 'boundaries shape: ', boundaries2.shape

        # seg_img = mat_data['GTcls']['Segmentation']

        # print 'seg image: ', seg_img  ## is an array
        # seg_img = seg_img.tolist()
        # print 'seg type: ', seg_img.dtype
        # print 'seg img 0:', seg_img[0]
        # print 'seg img shape: ', seg_img.shape  ## (1, 1) --> remove extract dim
        all_mask = seg_img[0][0]  # --> keep all masks
        uni_instance = np.unique(all_mask)
        print 'uni instance: ', uni_instance

        obj_id = mat_data['GTinst']['Categories']
        obj_id = obj_id[0][0]

        # print obj_id[0][0] ## 8
        # print obj_id[1][0] ## 9

        all_boxes = []
        for ind in range(1, len(uni_instance)):
            print '--------------------------'
            print 'current instance: ', ind
            # GET ONLY CURRENT MASK FOR CURRENT IND only current mask for current ind
            current_mask = np.copy(seg_img[0][0])  # --> keep all masks  # copy to not set reference
            print 'unique current mask: ', np.unique(current_mask)

            current_mask[current_mask != ind] = 0  # reset all values != current ind to background
            current_mask[current_mask == ind] = 1  # reset all values = current ind to binary (1)

            # visualization
            rgb = label_colours.take(current_mask, axis=0).astype('uint8')
            out_file = out_instance_vis_path + '/' + file_name + '_' + str(ind) + '.png'
            # out_file = os.path.join(out_instance_vis_path, file_name + '_' + str(ind) + '.png')
            # out_file = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/cls/' + file_name + '.png'
            sp.misc.toimage(rgb, cmin=0.0, cmax=255).save(out_file)

            # save bin mask
            seg_mask_path = out_instance_pkl_path + '/' + file_name + '_' + str(ind) + '_segmask.sm'
            # seg_mask_path = os.path.join(out_instance_pkl_path, file_name + '_' + str(ind) + '_segmask.sm')
            with open(seg_mask_path, 'wb') as f_seg_save:
                cPickle.dump(current_mask, f_seg_save, cPickle.HIGHEST_PROTOCOL)

            # get bbox coordinates
            [r, c] = np.where(all_mask == ind)
            x1 = np.min(c)
            x2 = np.max(c)
            y1 = np.min(r)
            y2 = np.max(r)

            print 'x1: ', x1
            print 'y1: ', y1

            print 'x2: ', x2
            print 'y2: ', y2

            # GET object id
            current_obj_id = obj_id[ind - 1][0]
            print 'current object id: ', current_obj_id

            current_bbox = [current_obj_id, x1, y1, x2, y2]

            all_boxes.append(current_bbox)


            #         # GET BOUNDARY OF CURRENT MASK - NO USE
            #         current_list_bound = boundaries[0][0][ind-1][0]
            #         print 'current_list_bound', current_list_bound
            #         print 'current list bound shape: ', current_list_bound.shape  # (width, height) of image
            #         # convert spare to dense matrix
            #         dense_bound = current_list_bound.todense()
            #         print 'dense list: ', dense_bound
            #         print 'dense list shape: ', dense_bound.shape
            #         # find top left coordinate
            #
            #
            #         #print 'curret list 1: ', current_list_bound[1,:]
            #         xmin = 500
            #         ymin = 400
            #         xmax = 0
            #         ymax = 0
            #
            #         n_row, n_column = dense_bound.shape
            #         dense_bound = np.asarray(dense_bound)   # convert matrix to array
            # #         for i in range(n_row-1):
            # #             for j in range(n_column-1):
            # #                 if dense_bound[i][j] == 1:
            # #                     if ymin < i:
            # #                         ymin = i
            # #                     if xmin < j:
            # #                         xmin = j
            #
            #         # find first "min" row
            #         row_min = 0
            #         for i in range(n_row-1):
            #             for j in range(n_column-1):
            #                 if dense_bound[i][j] == 1:
            #                     row_min = i
            #                     break
            #
            #         # find row max
            #         for i in range(n_row-1, 0, -1):
            #             for j in range(n_column-1):
            #                 if dense_bound[i][j] == 1:
            #                     row_max = i
            #                     break
            #
            #         print 'row min: ', row_min
            #         print 'row max: ', row_max
            #
            #
            #                     #break
            # #
            # #         print 'xmin: ', xmin
            # #         print 'ymin: ', ymin
            #
            #
            #         rgb = label_colours.take(dense_bound, axis=0).astype('uint8')
            #         out_file = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/ins/' + file_name + '_' + str(ind) + '_bbox' + '.png'
            #         #out_file = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/cls/' + file_name + '.png'
            #         sp.misc.toimage(rgb, cmin=0.0, cmax=255).save(out_file)

        # write to XML
        xmlpath = out_xml_path + '/' + file_name + '.xml'
        im_name = file_name + '.jpg'
        im_h, im_w = all_mask.shape
        im_c = 3

        writeXML(xmlpath, im_name, im_h, im_w, im_c, all_boxes)




        # visualize all mask
        #     rgb = label_colours.take(all_mask, axis=0).astype('uint8')
        #     out_file = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/ins/' + file_name + '.png'
        #     #out_file = '/home/anguyen/workspace/paper_src/2018.icra.sds.source/output/cls/' + file_name + '.png'
        #     sp.misc.toimage(rgb, cmin=0.0, cmax=255).save(out_file)

        # print 'seg img: ', seg_img

        ## debug
        # break


def main():
    train_txt_file = os.path.join(pascal_instance_folder, 'VOCSDS_FROM_FCIS', 'ImageSets', 'Main', 'train.txt')
    list_train = list(open(train_txt_file, 'r'))
    list_train = [x.strip() for x in list_train]
    print 'len list train: ', len(list_train)
    # print 'list train: ', list_train

    val_txt_file = os.path.join(pascal_instance_folder, 'VOCSDS_FROM_FCIS', 'ImageSets', 'Main', 'val.txt')
    list_val = list(open(val_txt_file, 'r'))
    list_val = [x.strip() for x in list_val]
    print 'len list val: ', len(list_val)
    # print 'list val: ', list_val


    parse_instance(list_train, out_train_xml_path, out_train_instance_pkl_path, out_train_instance_vis_path)
    parse_instance(list_val, out_val_xml_path, out_val_instance_pkl_path, out_val_instance_vis_path)


if __name__ == '__main__':
    main()
    print 'ALL DONE'
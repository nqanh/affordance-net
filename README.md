## [AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection](https://arxiv.org/pdf/1709.07326.pdf)
By Thanh-Toan Do\*, Anh Nguyen\*, Ian Reid (\* equal contribution)

![affordance-net](https://raw.githubusercontent.com/nqanh/affordance-net/master/tools/temp_output/iit_aff_dataset.jpg "affordance-net")

### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Demo](#demo)
4. [Training](#training)
5. [Notes](#notes)


### Requirements

1. Caffe
	- Install Caffe: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html).
	- Caffe must be built with support for Python layers.

2. Hardware
	- To train a full AffordanceNet, you'll need a GPU with ~11GB (e.g. Titan, K20, K40, Tesla, ...).
	- To test a full AffordanceNet, you'll need ~6GB GPU.

3. [Optional] For robotic demo
	- [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu)
	- [rospy](http://wiki.ros.org/rospy)
	- [OpenNI](https://github.com/OpenNI/OpenNI)
	- [PrimeSensor](https://github.com/PrimeSense/Sensor)


### Installation

1. Clone the AffordanceNet repository into your `$AffordanceNet_ROOT` folder.
	
	
2. Build `Caffe` and `pycaffe`:
	- `cd $AffordanceNet_ROOT/caffe-affordance-net`
    - `# Now follow the Caffe installation instructions: http://caffe.berkeleyvision.org/installation.html`
    - `# If you're experienced with Caffe and have all of the requirements installed and your Makefile.config in place, then simply do:`
    - `make -j8 && make pycaffe`
     

3. Build the Cython modules:
    - `cd $AffordanceNet_ROOT/lib`
    - `make`


4. Download pretrained weights ([Google Drive](https://drive.google.com/file/d/0Bx3H_TbKFPCjNlMtSGJlQ0dxVzQ/view?usp=sharing), [One Drive](https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/nqanh_mso_hcmus_edu_vn/ETD6q64-L1lCgtNEryA42NwBNM9vNoyE8QyxAYzgt8NqnA?e=uRCxPg)). This weight is trained on the training set of the [IIT-AFF dataset](https://sites.google.com/site/iitaffdataset/):
    - Extract the file you downloaded to `$AffordanceNet_ROOT`
    - Make sure you have the caffemodel file like this: `'$AffordanceNet_ROOT/pretrained/AffordanceNet_200K.caffemodel`

	
### Demo

After successfully completing installation, you'll be ready to run the demo. 

0. Export pycaffe path:
	- `export PYTHONPATH=$AffordanceNet_ROOT/caffe-affordance-net/python:$PYTHONPATH`

1. Demo on static images:
	- `cd $AffordanceNet_ROOT/tools`
	- `python demo_img.py`
	- You should see the detected objects and their affordances.
	
2. (Optional) Demo on depth camera (such as Asus Xtion):
	- With AffordanceNet and the depth camera, you can easily select the interested object and its affordances for robotic applications such as grasping, pouring, etc.
	- First, launch your depth camera with ROS, OpenNI, etc.
	- `cd $AffordanceNet_ROOT/tools`
	- `python demo_asus.py`
	- You may want to change the object id and/or affordance id (line `380`, `381` in `demo_asus.py`). Currently, we select the `bottle` and its `grasp` affordance.
	- The 3D grasp pose can be visualized with [rviz](http://wiki.ros.org/rviz). You should see something like this: 
	![affordance-net-asus](https://raw.githubusercontent.com/nqanh/affordance-net/master/tools/temp_output/asus_affordance_net_demo.jpg "affordance-net-asus")
	
### Training

1. We train AffordanceNet on [IIT-AFF dataset](https://sites.google.com/site/iitaffdataset/)
	- We need to format IIT-AFF dataset as in Pascal-VOC dataset for training.
	- For your convinience, we did it for you. Just download this file ([Google Drive](https://drive.google.com/file/d/0Bx3H_TbKFPCjV09MbkxGX0k1ZEU/view?usp=sharing), [One Drive](https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/nqanh_mso_hcmus_edu_vn/EXQok71Y2kFAmhaabY2TQO8BFIO1AqqH5GcMOfPqgn_q2g?e=7rH3Kd)) and extract it into your `$AffordanceNet_ROOT` folder.
	- The extracted folder should contain three sub-folders: `$AffordanceNet_ROOT/data/cache`, `$AffordanceNet_ROOT/data/imagenet_models`, and `$AffordanceNet_ROOT/data/VOCdevkit2012` .

2. Train AffordanceNet:
	- `cd $AffordanceNet_ROOT`
	- `./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]`
	- e.g.: `./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc`
	- We use `pascal_voc` alias although we're training using the IIT-AFF dataset.



### Notes
1. AffordanceNet vs. Mask-RCNN: AffordanceNet can be considered as a general version of Mask-RCNN when we have multiple classes inside each instance.
2. The current network achitecture is slightly diffrent from the paper, but it achieves the same accuracy.
3. Train AffordanceNet on your data:
	- Format your images as in Pascal-VOC dataset (as in `$AffordanceNet_ROOT/data/VOCdevkit2012` folder).
	- Prepare the affordance masks (as in `$AffordanceNet_ROOT/data/cache` folder): For each object in the image, we need to create a mask and save as a .sm file. See `$AffordanceNet_ROOT/utils` for details.


### Citing AffordanceNet

If you find AffordanceNet useful in your research, please consider citing:

	@inproceedings{AffordanceNet18,
	  title={AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection},
	  author={Do, Thanh-Toan and Nguyen, Anh and Reid, Ian},
	  booktitle={International Conference on Robotics and Automation (ICRA)},
	  year={2018}
	}


If you use [IIT-AFF dataset](https://sites.google.com/site/iitaffdataset/), please consider citing:

	@inproceedings{Nguyen17,
	  title={Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields},
	  author={Nguyen, Anh and Kanoulas, Dimitrios and Caldwell, Darwin G and Tsagarakis, Nikos G},
	  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	  year={2017},
	}


### License
MIT License

### Acknowledgement
This repo used a lot of source code from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn)


### Contact
If you have any questions or comments, please send us an email: `thanh-toan.do@adelaide.edu.au` and `anh.nguyen@iit.it`


## [AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection](https://arxiv.org/pdf/1709.07326.pdf)
By Thanh-Toan Do\*, Anh Nguyen\*, Ian Reid, Darwin G. Caldwell, Nikos G. Tsagarakis (\* equal contribution)

![alt text](https://github.com/nqanh/affordance-net/tree/master/tools/temp_output/iit_aff_dataset.jpg "affordance-net")

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
	- Smaller net will be avalable soon.

3. [Optional] For robotic demo
	- [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu)
	- [rospy](http://wiki.ros.org/rospy)
	- [OpenNI](https://github.com/OpenNI/OpenNI)
	- [PrimeSensor](https://github.com/PrimeSense/Sensor)


### Installation

1. Clone the AffordanceNet repository into `$AffordanceNet_ROOT` folder:
	- `git clone https://github.com/nqanh/affordance-net.git`
	
	
2. Build `Caffe` and `pycaffe`:
	- `cd $AffordanceNet_ROOT/caffe-affordance-net`
    - `# Now follow the Caffe installation instructions: http://caffe.berkeleyvision.org/installation.html`
    - `# If you're experienced with Caffe and have all of the requirements installed and your Makefile.config in place, then simply do:`
    - `make -j8 && make pycaffe`
     

3. Build the Cython modules:
    - `cd $AffordanceNet_ROOT/lib`
    - `make`


4. Download [pretrained weights](https://drive.google.com/file/d/0Bx3H_TbKFPCjNlMtSGJlQ0dxVzQ/view?usp=sharing):
    - Extract the file you downloaded to `$AffordanceNet_ROOT`
    - Make sure you have the caffemodel file like this: `'$AffordanceNet_ROOT/pretrained/AffordanceNet_200K.caffemodel`

	
### Demo

After successfully completing installation, you'll be ready to run the demo. 

0. Export pycaffe path:
	- `export PYTHONPATH=$AffordanceNet_ROOT/caffe-affordance-net/python:$PYTHONPATH`

1. Demo on static images:
	- `cd $AffordanceNet_ROOT`
	- `./tools/demo_img.py`
	- You should see the detected objects and their affordances:
	
2. (Optional - Comming soon) Demo on depth camera (such as Asus Xtion):
	- Launch your camera with ROS, OpenNI, etc.
	- `cd $AffordanceNet_ROOT`
	- `./tools/demo_asus.py`
	- You may want to change the camera topic name (e.g. `/camera/left/image_raw`) to yours.
	
### Training

1. We train AffordanceNet on [IIT-AFF dataset](https://sites.google.com/site/iitaffdataset/)
	- We need to format IIT-AFF dataset as in Pascal-VOC dataset for training.
	- For your convinience, we did it for you. Just download [this file](https://drive.google.com/file/d/0Bx3H_TbKFPCjV09MbkxGX0k1ZEU/view?usp=sharing) and extract it into your `$AffordanceNet_ROOT` folder.
	- The extracted folder should contain three sub-folders: `$AffordanceNet_ROOT/data/cache`, `$AffordanceNet_ROOT/data/imagenet_models`, and `$AffordanceNet_ROOT/data/VOCdevkit2012` .

2. Train AffordanceNet:
	- `cd $AffordanceNet_ROOT`
	- `./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]`
	- e.g.: `./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc`
	- We use `pascal_voc` alias although we're training using the IIT-AFF dataset.



### Notes
1. AffordanceNet vs. Mask-RCNN: AffordanceNet can be considered as a general version of Mask-RCNN when we have multiple classes inside each instance.
2. The current network achitecture is slightly diffrent from the paper, but it achieves the same accuracy.
	


### Citing AffordanceNet

If you find AffordanceNet useful in your research, please consider citing:

	@article{AffordanceNet17,
	  title={AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection},
	  author={Do, Thanh-Toan and Nguyen, Anh and Reid, Ian and Caldwell, Darwin G and Tsagarakis, Nikos G},
	  journal={arXiv:1709.07326},
	  year={2017}
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


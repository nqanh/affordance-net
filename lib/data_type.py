import numpy as np
import scipy.sparse
import cPickle
from PIL import Image
import cv2
from pycocotools import mask as COCOmask

def process_str(arg1="10", arg2="20"):
    str = "{}x{}".format(arg1,arg2)
    print str

process_str("50","100")
strarr = "VGG_VOC0712_{}".format("SSD")
print (str)

a = np.ones((2,2))  # Create an array of all zeros
print a

mbox_source_layers = ['res3b3_relu', 'res5c_relu', 'res5c_relu/conv1_2', 'res5c_relu/conv2_2', 'res5c_relu/conv3_2','pool6']

for strarr in mbox_source_layers:
    print strarr

print "\n"

#for i in xrange(0,len(mbox_source_layers)):
for i in range(len(mbox_source_layers)-1, 0, -1):
#for i in range(len(mbox_source_layers)-1, len(mbox_source_layers)-2, -1):
    print mbox_source_layers[i]
    print mbox_source_layers[i-1]

print "test min:"
a = 5
b = 6

b = min(a,b)
print (str(b) + "test num2str")

for i in range(len(mbox_source_layers)-1, 0, -1):
    print i


for i in range(0,5):
    print ("i=" + str(i))
    maxsize = []
    print ("length maxsize = " + str(len(maxsize)))
    if maxsize:
        print ("co maxsize")

print "test max and arg_max"

nrow = 3 #number of boxes
ncol = 2 #number of gt boxes
a = np.zeros((nrow,ncol),dtype=np.float32)
#a = np.random.rand(nrow,ncol)
for i in xrange(nrow):
    for j in xrange(ncol):
        a[i][j] = np.random.rand(1,1)

print a
argmaxes = a.argmax(axis=1) #tra ve index cua cot chua gia tri max
print argmaxes
maxes = a.max(axis=1) #tra ve gia tri max tren tung hang
print maxes

#I = np.where(maxes > 0)
#print I

I = np.where(maxes > 0)[0] # lay ra index cua nhung phan tu > 0 trong mang maxes
print I


print a.shape[0] #number of rows of matrix
print a.shape[1] #number of cols of matrix
'''
a = [[ 0.6588949   0.21445625]
 [ 0.2379213   0.34548581]
 [ 0.11590031  0.76919836]]
argmaxes = [0 1 1]
maxes = [ 0.6588949 (max of row 1)   0.34548581 (max of row 2)  0.76919836 (max of row 3)]
I = [0 1 2]
'''

print "test shift"
width = 38
height = 63
feat_stride = 16
shift_x = np.arange(0,width) * feat_stride
shift_y = np.arange(0,height) * feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()
#print shift_x
#print shift_y


print shifts.shape[0]
print shifts

print 2**np.arange(3,6) # = 2^[3 4 5]

a = np.random.rand(2, 3)
print  a

a = np.zeros((2,3), dtype=np.bool_)
a[0][0] = 1
a[0][1] = 0
print a
print a.dtype

a1 = a.astype('uint8')
print a1
#a = a.astype(int)


#print a.ravel() #convert 2D to 1D array
b = np.random.rand(2,3)
print "data type of matrix b:"
print b.dtype
print "matrix b:"
print b
b = scipy.sparse.csr_matrix(b)
print "matrix b:"
print b

c = b.toarray()
print c

a = c
a = a.astype(bool)
print a.dtype
print a

print "read mask file"

filename = '/home/tdo/Software/FRCN_ROOT/data/cache/seg_mask_coco_gt/375342_1_segmask.sm'
with open(filename, 'rb') as f:
    seg_mask = cPickle.load(f)
seg_mask = (seg_mask*255).astype('uint8')
img = Image.fromarray(seg_mask).convert('RGB')

print "image size"
print img.size
# img.show()

b = np.random.rand(2,3)
print b
#print b[:, 0:2] #print tat ca cac hang. cot 0 va cot 1
#print "value of first row"
print b[:,:2] #print tat ca cac hang, bo cot thu 2 (chi lay cot 0 va cot 1)

print b.shape[1]
# c = 1000
# print c

b = [1, 2, 3, 4]
print len(b)

print "check 2D array"
flipped = [[False]]
print flipped[0][0]

print "check greater / smaller operator"
c = [[1, 2, 3, 4],
     [-1, -2, -3, -4]]
c = np.asarray(c)
d = np.ravel(c)
# print d
print d.reshape(2,4)
# print d
#ind = d[(d >=-2)  & (d<=0)]
# ind = ((c >=-2) & (c<=0))
# print ind
#
# c[ind] = 100
# print c

# c[(c >=-2) & (c<=0)] = 100
print c
idx = (c >=-2) & (c<=0)
print "=====================================================idx: "
print idx

print "check round function"
d = 0.6
print round(d)

print "check all zero array"
a = [0, 0, 0, 1]
a = np.asarray(a)
allzero = np.any(a)
print allzero

#a = np.random.rand(2,3)
a = np.ones((2,3), dtype=np.float32)
print "before scale"
print a
im_scale = 2
a = cv2.resize(a, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
print "after scale"
print a

a = [0, 0, 0, 1]
b = np.asarray(a)
print len(a) #=4
print b.size #=4
print len(b) #=4

a = [0, 0, 0, 1]
b = [2, 2, 2]
print np.append(a,b)

print round(3.156)


a = np.zeros((2,4), dtype=np.float32)
print a.shape
print a

b = np.zeros((1,4), dtype=np.uint8)
b[:,:] = [1, 2, 3, 4]
a[1, :] = b
print a #a van la kieu float32
print b #b van la kieu uint8

s = 'coco'
if s == 'coco':
    print "hello coco"

a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
print a*b

m = np.maximum(0,a)
print m

mlist = []
mlist.append([0])
mlist.append([1, 2, 3])
print mlist

a = np.random.rand(4,3,4)
b = np.ones((4, 4), dtype=np.float32)
b[:,:] = a[:,1,:]
print "values of b: "
print b

print ("size of b: " + str(b.shape) + "\n")

print b.max()


print np.min((np.max((2.0, 1)),4.0))

CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush')

print ("number of class")
print len(CLASSES)
print CLASSES[0]

a = np.random.rand(4,4)
print ("matrix a: ")
print a
print ("row 0-2, col 0-2:")
print a[0:2,0:2] # chi lay 0, 1 ma thoi

b = np.random.rand(4,1)
b[0:3,0:1] = np.tile(1, [3,1])
print "matrix b: "
print b

c = np.tile(b, [1, 2])
print "matrix c"
print c

a = np.array( [[1, 2, 3]],dtype=np.float32)
print a[0,2]

a = np.zeros((1,5), dtype=np.float32)
print a.shape

a = np.arange(255, 15, -50)
print a


num_images = 2
num_classes = 4

all_boxes = [[[] for _ in xrange(num_images)]
 for _ in xrange(num_classes)]

#==> allboxes = [[[], []] --> c0-{i0,i1}, [[], []]--> c1-{i0,i1}, [[], []]--> c2-{i0,i1}, [[], []]]--> c3-{i0,i1}

arle = {}
for n in xrange(num_images):
    arle[n] = {}

# print arle

a = np.ones((5, 5), dtype=np.uint8, order='F')
# a = np.ones((5,5), dtype=np.uint8)
# a = a.reshape(-1)
# a = np.expand_dims(a, axis=0)
rle_a = COCOmask.encode(a)


b = np.ones((5,10), dtype=np.uint8)
b = b.reshape(-1)
b = np.expand_dims(b, axis=0)
rle_b = COCOmask.encode(b)

# arle[0] = rle_a
arle[0] = np.vstack([arle[0], rle_a])
arle[0] = np.vstack([arle[0], rle_b])
print "arle"
print arle

all_rles = [[{} for _ in xrange(num_images)]
             for _ in xrange(num_classes)]
print "init all_rles"
print all_rles

# print "all_rles[0][0]"
# print all_rles[0][0]

print "len(all_rles)"
print len(all_rles)

# if len (all_rles[0][0]):
print "len(all_rles[0][0])"
print len(all_rles[0][0])

if len(all_rles[0][0]) == 0:
    print "if"
    all_rles[0][0] = np.hstack([rle_a])
    all_rles[0][0] = np.hstack([all_rles[0][0], rle_b])
else:
    print "else"
    all_rles[0][0] = np.hstack([all_rles[0][0], rle_a])

if len(all_rles[0][0]):
    print len(all_rles[0][0])
    print "co conga2"
# all_rles = {}
# for n in xrange(num_images):
#     for c in xrange(1, num_classes):  # ignore bg class --> [[]]
#         all_rles[c][n] = {}
#

# all_rles_per_class = all_rles[0]
# des_rles_per_class_per_img =  all_rles_per_class[0]
# print des_rles_per_class_per_img

#print "all_rles[0]"
print all_rles[0]
c = 0
a1 = all_rles[c] #class c
print "a1"
print a1
i = 0
a2 = a1[i] #image i
print "len(a2)"
print len(a2)
print "a2"
print a2
if len(a2) == 0:
    print "conket"

print "a2.shape[0]"
print a2.shape[0]
print a2[0]
# print "rle_a"
# print rle_a

results = []
results.extend(
    [{'image_id': 1,
      'category_id': 2,
      'bbox': [3, 3, 3, 3],
      'score': 4} for k in xrange(2)]) #-->hstack
print "results"
print results
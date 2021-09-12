import rospy 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class ImageIO:
    def __init__(self):
        self.bridge = CvBridge()
        self.asus_rgb_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.asus_rgb_callback)
        self.asus_dep_sub = rospy.Subscriber("/camera/depth/image_rect", Image, self.asus_dep_callback)
        self.asus_cam_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.asus_cam_callback) 
        self.asus_rgb_img = None
        self.asus_dep_img = None
        
        self.asus_K  = None
        self.asus_fx = None 
        self.asus_fy = None 
        self.asus_cx = None 
        self.asus_cy = None   
        self.asus_width = None
        self.asus_height = None
            
    def asus_rgb_callback(self, data):
        try:
            self.asus_rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")            
        except CvBridgeError as e:
            print(e)
    
    def asus_dep_callback(self, dep_msg):
        try:
            self.asus_dep_img = self.bridge.imgmsg_to_cv2(dep_msg, 'passthrough')
        
        except CvBridgeError as e:
            print (e)
            
    def asus_cam_callback(self, cam_msg):
        if (self.asus_K == None):
            self.asus_K = cam_msg.K
#             self.asus_K = [550.6944711701044, 0.0, 325.91239272527577, 0.0, 553.678494137733, 233.01009570210894, 0.0, 0.0, 1.0]
#             self.asus_K = [537.719521812601, 0.0, 317.7514539076608, 0.0, 532.663175739311, 226.92788210884436, 0.0, 0.0, 1.0]
            self.asus_fx = self.asus_K[0]
            self.asus_cx = self.asus_K[2]
            self.asus_fy = self.asus_K[4]
            self.asus_cy = self.asus_K[5]
            self.asus_width = cam_msg.width
            self.asus_height = cam_msg.height            
        else:
            self.asus_cam_sub.unregister()
            

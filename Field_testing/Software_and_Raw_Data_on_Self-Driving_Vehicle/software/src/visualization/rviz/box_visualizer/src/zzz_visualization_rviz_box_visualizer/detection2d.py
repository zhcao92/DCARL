from easydict import EasyDict
import rospy
import math
import cv2
import numpy as np
import cv_bridge

from sensor_msgs.msg import Image
from zzz_visualization_rviz_box_visualizer.utils import parse_color

class DetectionBox2DVisualizer:
    def __init__(self, **params):
        self._marker_id = 0
        self._params = EasyDict(params)
        self._image = np.zeros((800, 600, 3), np.uint8)
        self._bridge = cv_bridge.CvBridge()

    def receive_image(self, img_msg):
        self._image = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")

    def add_labels(self, in_objects, in_img):
        for obj in in_objects.detections:
            # TODO: Implement this
            pass
        return in_img

    def add_boxes(self, in_objects, in_img):
        for obj in in_objects.detections:
            in_img = cv2.rectangle(in_img,
                (int(obj.bbox.pose.x - obj.bbox.dimension.length_x/2),
                 int(obj.bbox.pose.y - obj.bbox.dimension.length_y/2)), 
                (int(obj.bbox.pose.x + obj.bbox.dimension.length_x/2),
                 int(obj.bbox.pose.y + obj.bbox.dimension.length_y/2)), self._params.box_color, 2) # thickness
        return in_img

    def visualize(self, msg):
        self._marker_id = 0

        image = self._image
        image = self.add_boxes(msg, image)
        image = self.add_labels(msg, image)

        return self._bridge.cv2_to_imgmsg(image, encoding="passthrough")

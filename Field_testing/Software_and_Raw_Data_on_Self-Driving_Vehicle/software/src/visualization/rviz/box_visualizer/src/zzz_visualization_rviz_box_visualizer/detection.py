from easydict import EasyDict
import rospy
import math

from visualization_msgs.msg import Marker, MarkerArray
from zzz_visualization_rviz_box_visualizer.utils import parse_color

class DetectionBoxVisualizer:
    def __init__(self, **params):
        self._marker_id = 0
        self._params = EasyDict(params)

        if isinstance(self._params.label_color, list):
            self._params.label_color = parse_color(self._params.label_color)
        if isinstance(self._params.box_color, list):
            self._params.box_color = parse_color(self._params.box_color)
        if isinstance(self._params.centroid_color, list):
            self._params.centroid_color = parse_color(self._params.centroid_color)

    def addLabels(self, in_objects, out_markers):
        for obj in in_objects.detections:
            label_marker = Marker()
            label_marker.lifetime = rospy.Duration(self._params.marker_lifetime)

            # Message headers
            label_marker.header = in_objects.header
            label_marker.action = Marker.ADD
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.ns = self._params.marker_namespace + "/labels"
            label_marker.id = self._marker_id
            self._marker_id += 1

            # Marker properties
            label_marker.scale.x = self._params.label_scale
            label_marker.scale.y = self._params.label_scale
            label_marker.scale.z = self._params.label_scale
            label_marker.color = self._params.label_color

            if len(obj.classes) > 0: # Use classification name if presented
                label_marker.text = "[" + str(obj.classes[0].classid) + "]"

            x = obj.bbox.pose.pose.position.x
            y = obj.bbox.pose.pose.position.y
            label_marker.text += "%.2f m" % math.sqrt(x*x + y*y)
            if obj.comments:
                label_marker.text += "\n" + obj.comments

            label_marker.pose.position.x = x
            label_marker.pose.position.y = y
            label_marker.pose.position.z = self._params.label_height
            
            out_markers.markers.append(label_marker)

    def addBoxes(self, in_objects, out_markers):
        for obj in in_objects.detections:
            box = Marker()
            box.lifetime = rospy.Duration(self._params.marker_lifetime)

            # Message headers
            box.header = in_objects.header
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.ns = self._params.marker_namespace + "/boxs"
            box.id = self._marker_id
            self._marker_id += 1

            # Marker properties
            box.color = self._params.box_color

            box.scale.x = obj.bbox.dimension.length_x
            box.scale.y = obj.bbox.dimension.length_y
            box.scale.z = obj.bbox.dimension.length_z
            box.pose.position = obj.bbox.pose.pose.position
            box.pose.orientation = obj.bbox.pose.pose.orientation

            # Skip large boxes to prevent messy visualization
            if box.scale.x > self._params.box_max_size or box.scale.y > self._params.box_max_size or box.scale.z > self._params.box_max_size:
                rospy.logdebug("Object skipped with size %.2f x %.2f x %.2f!",
                    box.scale.x, box.scale.y, box.scale.z)
                continue

            out_markers.markers.append(box)

    def addCentroids(self, in_objects, out_markers):
        for obj in in_objects.detections:
            centroid_marker = Marker()
            centroid_marker.lifetime = rospy.Duration(self._params.marker_lifetime)

            # Message headers
            centroid_marker.header = in_objects.header
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.ns = self._params.marker_namespace + "/centroids"
            centroid_marker.id = self._marker_id
            self._marker_id += 1

            # Marker properties
            centroid_marker.scale.x = self._params.centroid_scale
            centroid_marker.scale.y = self._params.centroid_scale
            centroid_marker.scale.z = self._params.centroid_scale
            centroid_marker.color = self._params.centroid_color

            centroid_marker.pose = obj.bbox.pose.pose

            out_markers.markers.append(centroid_marker)

    def visualize(self, msg):
        markers = MarkerArray()
        self._marker_id = 0

        self.addBoxes(msg, markers)
        self.addCentroids(msg, markers)
        self.addLabels(msg, markers)

        return markers        

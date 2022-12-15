from easydict import EasyDict
import rospy
import numpy as np

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from zzz_visualization_rviz_box_visualizer.utils import parse_color

class TrackingBoxVisualizer:
    def __init__(self, **params):
        self._params = EasyDict(params)
        self._tracker_set = set()
        
        if isinstance(self._params.label_color, list):
            self._params.label_color = parse_color(self._params.label_color)
        if isinstance(self._params.box_color, list):
            self._params.box_color = parse_color(self._params.box_color)
        if isinstance(self._params.centroid_color, list):
            self._params.centroid_color = parse_color(self._params.centroid_color)
        if isinstance(self._params.arrow_color, list):
            self._params.arrow_color = parse_color(self._params.arrow_color)

    def addLabels(self, in_objects, out_markers):
        for obj in in_objects.targets:
            label_marker = Marker()

            # Message headers
            label_marker.header = in_objects.header
            label_marker.action = Marker.MODIFY if obj.uid in self._tracker_set else Marker.ADD
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.ns = self._params.marker_namespace + "/labels"
            label_marker.id = obj.uid

            # Marker properties
            label_marker.scale.x = self._params.label_scale
            label_marker.scale.y = self._params.label_scale
            label_marker.scale.z = self._params.label_scale
            label_marker.color = self._params.label_color

            label_marker.text = "#%d: " % obj.uid
            if len(obj.classes) > 0: # Use classification name if presented
                label_marker.text += "[" + str(obj.classes[0].classid) + "]"

            x = obj.bbox.pose.pose.position.x
            y = obj.bbox.pose.pose.position.y
            z = obj.bbox.pose.pose.position.z
            vx = obj.twist.twist.linear.x
            vy = obj.twist.twist.linear.y
            vz = obj.twist.twist.linear.z
            label_marker.text += "%.2f m, %.2f m/s" % (np.sqrt(x*x + y*y + z*z), np.sqrt(vx*vx + vy*vy + vz*vz))
            if obj.comments:
                label_marker.text += "\n" + obj.comments

            label_marker.pose.position.x = x
            label_marker.pose.position.y = y
            label_marker.pose.position.z = self._params.label_height
            
            out_markers.markers.append(label_marker)

    def addBoxes(self, in_objects, out_markers):
        for obj in in_objects.targets:
            box = Marker()

            # Message headers
            box.header = in_objects.header
            box.type = Marker.CUBE
            box.action = Marker.MODIFY if obj.uid in self._tracker_set else Marker.ADD
            box.ns = self._params.marker_namespace + "/boxs"
            box.id = obj.uid

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

            box_str = str(box)
            if 'nan' in box_str or 'inf' in box_str:
                print("----------------------------------")
                print(box_str)
            out_markers.markers.append(box)

    def addCentroids(self, in_objects, out_markers):
        for obj in in_objects.targets:
            centroid_marker = Marker()

            # Message headers
            centroid_marker.header = in_objects.header
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.MODIFY if obj.uid in self._tracker_set else Marker.ADD
            centroid_marker.ns = self._params.marker_namespace + "/centroids"
            centroid_marker.id = obj.uid

            # Marker properties
            centroid_marker.scale.x = self._params.centroid_scale
            centroid_marker.scale.y = self._params.centroid_scale
            centroid_marker.scale.z = self._params.centroid_scale
            centroid_marker.color = self._params.centroid_color

            centroid_marker.pose = obj.bbox.pose.pose

            out_markers.markers.append(centroid_marker)

    def addArrow(self, in_objects, out_markers):
        '''
        This function shows the velocity direction of the object
        '''
        for obj in in_objects.targets:
            arrow_marker = Marker()

            # Message headers
            arrow_marker.header = in_objects.header
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.MODIFY if obj.uid in self._tracker_set else Marker.ADD
            arrow_marker.ns = self._params.marker_namespace + "/arrows"
            arrow_marker.id = obj.uid

            # Marker properties
            arrow_marker.color = self._params.arrow_color
            arrow_marker.scale.x = self._params.arrow_width
            arrow_marker.scale.y = 0.4
            arrow_marker.scale.z = 0.5

            arrow_marker.pose.position = obj.bbox.pose.pose.position

            vel = np.array([obj.twist.twist.linear.x, obj.twist.twist.linear.y, obj.twist.twist.linear.z])
            vel_len = np.linalg.norm(vel)
            if vel_len > 0:
                arrow_marker.points.append(Point())
                p1 = Point()
                p1.x = obj.twist.twist.linear.x * self._params.arrow_speed_scale
                p1.y = obj.twist.twist.linear.y * self._params.arrow_speed_scale
                p1.z = obj.twist.twist.linear.z * self._params.arrow_speed_scale
                arrow_marker.points.append(p1)
            else:
                # No arrow is shown if the velocity is zero
                continue

            out_markers.markers.append(arrow_marker)

    def deleteMarkers(self, id_set, out_markers):
        for obj_id in id_set:
            marker_del = Marker()
            marker_del.action = Marker.DELETE
            marker_del.type = Marker.TEXT_VIEW_FACING
            marker_del.ns = self._params.marker_namespace + "/labels"
            marker_del.id = obj_id
            out_markers.markers.append(marker_del)

            marker_del = Marker()
            marker_del.action = Marker.DELETE
            marker_del.type = Marker.CUBE
            marker_del.ns = self._params.marker_namespace + "/boxs"
            marker_del.id = obj_id
            out_markers.markers.append(marker_del)
            
            marker_del = Marker()
            marker_del.action = Marker.DELETE
            marker_del.type = Marker.SPHERE
            marker_del.ns = self._params.marker_namespace + "/centroids"
            marker_del.id = obj_id
            out_markers.markers.append(marker_del)
            
            marker_del = Marker()
            marker_del.action = Marker.DELETE
            marker_del.type = Marker.ARROW
            marker_del.ns = self._params.marker_namespace + "/arrows"
            marker_del.id = obj_id
            out_markers.markers.append(marker_del)


    def visualize(self, msg):
        markers = MarkerArray()

        # Remove old trackers
        new_set = set()
        for obj in msg.targets:
            new_set.add(obj.uid)
        self.deleteMarkers(self._tracker_set.difference(new_set), markers)

        # Add new trackers
        self.addBoxes(msg, markers)
        self.addCentroids(msg, markers)
        self.addLabels(msg, markers)
        self.addArrow(msg, markers)

        self._tracker_set = new_set
        return markers        

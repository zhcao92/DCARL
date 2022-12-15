import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped



class rviz_display():
    def __init__(self):
        self.candidates_trajectory = None
        self.prediciton_trajectory = None
        self.collision_circle = None
        self.kick_in_signal = None
    
    def put_trajectory_into_marker(self, fplist):
        if fplist is None or len(fplist)<1:
            return None
        try:
            tempmarkerarray = MarkerArray()
            count = 0

            for fp in fplist:
                tempmarker = Marker() 
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/decision"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.05
                tempmarker.color.r = 0.0
                tempmarker.color.g = 1.0
                tempmarker.color.b = 0.5
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(1.0)
                for t in range(len(fp.x)):
                    p = Point()
                    p.x = fp.x[t]
                    p.y = fp.y[t]
                    tempmarker.points.append(p)
                tempmarkerarray.markers.append(tempmarker)
                count = count + 1
            return tempmarkerarray
        except:
            return None

    def draw_circles(self, fp1, fp2, radius):
        if fp1 is None or fp2 is None:
            return None
        
        tempmarkerarray = MarkerArray()
        tempmarker = Marker() 
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.CYLINDER
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = radius
        tempmarker.scale.y = radius
        tempmarker.scale.z = radius
        tempmarker.color.r = 0.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 0.5
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(1.0)
        tempmarker.pose.position.x = fp1.x[0]
        tempmarker.pose.position.y = fp1.y[0]
        tempmarkerarray.markers.append(tempmarker)

        tempmarker = Marker() 
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 2
        tempmarker.type = Marker.CYLINDER
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = radius
        tempmarker.scale.y = radius
        tempmarker.scale.z = radius
        tempmarker.color.r = 1.0
        tempmarker.color.g = 0.0
        tempmarker.color.b = 0.5
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(1.0)
        tempmarker.pose.position.x = fp2.x[0]
        tempmarker.pose.position.y = fp2.y[0]
        tempmarkerarray.markers.append(tempmarker)

        return tempmarkerarray

    def draw_obs_circles(self, fplist, radius):
        if fplist is None:
            return None
        tempmarkerarray = MarkerArray()
        count = 0

        for fp in fplist:
            tempmarker = Marker() 
            tempmarker.header.frame_id = "map"
            tempmarker.header.stamp = rospy.Time.now()
            tempmarker.ns = "zzz/decision"
            tempmarker.id = count
            tempmarker.type = Marker.CYLINDER
            tempmarker.action = Marker.ADD
            tempmarker.scale.x = radius
            tempmarker.scale.y = radius
            tempmarker.scale.z = radius
            tempmarker.color.r = 1.0
            tempmarker.color.g = 0.0
            tempmarker.color.b = 0.5
            tempmarker.color.a = 0.5
            tempmarker.lifetime = rospy.Duration(1.0)
            tempmarker.lifetime = rospy.Duration(1.0)
            for t in range(len(fp.x)):
                tempmarker.pose.position.x = fp.x[0]
                tempmarker.pose.position.y = fp.y[0]
                tempmarkerarray.markers.append(tempmarker)
            count = count + 1
        return tempmarkerarray
    
    def draw_kick_in_path(self, path):
        tempmarkerarray = MarkerArray()
            
        tempmarker = Marker() 
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.LINE_STRIP
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = 0.3
        tempmarker.color.r = 1.0
        tempmarker.color.g = 0.0
        tempmarker.color.b = 0.0
        tempmarker.color.a = 1.0
        tempmarker.lifetime = rospy.Duration(1.0)
        for wp in path:
            p = Point()
            p.x = wp[0]
            p.y = wp[1]
            tempmarker.points.append(p)
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray

    def draw_rule_path(self, path):
        tempmarkerarray = MarkerArray()
            
        tempmarker = Marker() 
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/decision"
        tempmarker.id = 1
        tempmarker.type = Marker.LINE_STRIP
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = 0.3
        tempmarker.color.r = 0.0
        tempmarker.color.g = 0.0
        tempmarker.color.b = 1.0
        tempmarker.color.a = 1.0
        tempmarker.lifetime = rospy.Duration(1.0)
        for wp in path:
            p = Point()
            p.x = wp[0]
            p.y = wp[1]
            tempmarker.points.append(p)
        tempmarkerarray.markers.append(tempmarker)
        return tempmarkerarray

# common used functions

def convert_ndarray_to_pathmsg(path):
    msg = Path()
    for wp in path:
        pose = PoseStamped()
        pose.pose.position.x = wp[0]
        pose.pose.position.y = wp[1]
        msg.poses.append(pose)
    msg.header.frame_id = "map" 

    return msg


def convert_path_to_ndarray(path):
    point_list = [(point.position.x, point.position.y) for point in path]
    return np.array(point_list)

    
def convert_XY_to_pathmsg(XX,YY,path_id = 'map'):
    msg = Path()

    for i in range(1,len(XX)):
        pose = PoseStamped()
        pose.pose.position.x = XX[i]
        pose.pose.position.y = YY[i]
        
        msg.poses.append(pose)
    msg.header.frame_id = path_id 
    return msg


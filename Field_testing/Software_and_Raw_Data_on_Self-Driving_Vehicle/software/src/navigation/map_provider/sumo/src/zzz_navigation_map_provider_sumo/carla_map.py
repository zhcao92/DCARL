import os, sys
import io
import subprocess
from collections import deque
import tempfile
import math
import time

import rospy
import numpy as np
from zzz_navigation_msgs.msg import Lane, LanePoint, LaneBoundary, Map
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from geometry_msgs.msg import PoseStamped, Point32
from nav_msgs.msg import Path

class CarlaMap(object):
    def __init__(self, offset_x=0, offset_y=0):
        self._ego_vehicle_x = None # location in meters
        self._ego_vehicle_y = None

        self._offset_x = offset_x # map center offset in meters
        self._offset_y = offset_y

        self._ego_vehicle_x_buffer = 0.0
        self._ego_vehicle_y_buffer = 0.0

        self._reference_lane_list = deque(maxlen=20000)
        self._lanes_inside = []
        self._lanes_outside = []
        self.load_lanes() # load lanes

    def load_lanes(self):
        # inner_path = os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/carla_data/inner_loop.dat'
        # outer_path = os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/carla_data/outer_loop.dat'
        # # inner 1, outer 0
        # self._lanes.append(self.get_lane(np.loadtxt(outer_path, delimiter=',')))
        # self._lanes.append(self.get_lane(np.loadtxt(inner_path, delimiter=',')))

        
        self._lanes_inside.append(dense_polyline2d(np.array([[23.4,-91.234],[-36.214,-91.5]]),1))
        self._lanes_inside.append(dense_polyline2d(np.array([[-50.942,-71.59],[-51.1,-10.56]]),1))
        self._lanes_inside.append(dense_polyline2d(np.array([[-37.62,3.11],[15.03,3.37]]),1))
        self._lanes_inside.append(dense_polyline2d(np.array([[31.8,-14.33],[33.52,-74.71]]),1))

        self._lanes_outside.append(dense_polyline2d(np.array([[22.246,-94.584],[-37,-94.817]]),1))
        self._lanes_outside.append(dense_polyline2d(np.array([[-55,-71.85],[-54.35,-10.114]]),1))
        self._lanes_outside.append(dense_polyline2d(np.array([[-37.11,6.15],[15.03,6.16]]),1))
        self._lanes_outside.append(dense_polyline2d(np.array([[34.76,-14.98],[36.53,-74.61]]),1))

    def ego_road(self):

        dist_list = np.array([dist_from_point_to_polyline2d(
                self._ego_vehicle_x, -self._ego_vehicle_y,
                lane) for lane in self._lanes_inside])
        closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        return closest_lane

    def receive_new_pose(self, x, y):
        self._ego_vehicle_x_buffer = x
        self._ego_vehicle_y_buffer = y


    def update(self):
        self._ego_vehicle_x = self._ego_vehicle_x_buffer
        self._ego_vehicle_y = self._ego_vehicle_y_buffer
        if self.should_update_static_map():
            self.update_static_map()
            return self.static_local_map

        return None


    def should_update_static_map(self):
        '''
        Determine whether map updating is needed.
        '''
        return True

    def init_static_map(self):
        '''
        Generate null static map
        '''
        init_static_map = Map()
        init_static_map.in_junction = False
        init_static_map.exit_lane_index.append(1)
        return init_static_map


    def update_static_map(self):
        ''' 
        Update information in the static map if current location changed dramatically
        '''
        map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y

        self.static_local_map = self.init_static_map()  ## Return this one
        self.update_lane_list()
        

    def update_lane_list(self):
        '''
        Update lanes when a new road is encountered
        '''
        # map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y
        
        # Left is 0 
        self.static_local_map.in_junction = False # lane change
        ego_road = self.ego_road()
        self.static_local_map.lanes.append(self.get_lane(self._lanes_outside[ego_road]))
        self.static_local_map.lanes.append(self.get_lane(self._lanes_inside[ego_road]))
        # rospy.loginfo("### native map update lane0 - {}, lane1 - {}".format(
        #     self.static_local_map.lanes[0].central_path_points[2357], 
        #     self.static_local_map.lanes[1].central_path_points[2188]))


    def get_lane(self, central_points): # outside

        lane_wrapped = Lane()
        for wp in central_points: # TODO Consider ego pose where is the start and end point.
            point = LanePoint()
            x, y = wp[0], wp[1]   # Point XY
            #x, y = wp[1], wp[0]
            point.position.x = x
            point.position.y = -y
            lane_wrapped.central_path_points.append(point)

        return lane_wrapped



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

class ShougangMap(object):
    def __init__(self, offset_x=0, offset_y=0):
        self._ego_vehicle_x = None # location in meters
        self._ego_vehicle_y = None

        self._offset_x = offset_x # map center offset in meters
        self._offset_y = offset_y

        self._ego_vehicle_x_buffer = 0.0
        self._ego_vehicle_y_buffer = 0.0

        self._reference_lane_list = deque(maxlen=20000)
        self._lanes = []
        self.load_lanes() # load lanes
        self._next_road_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]

        self._current_road_id = -1
        self._next_road_id = -1

    def load_lanes(self):
        
        road_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_0.txt',delimiter=',')
        road_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_1.txt',delimiter=',')
        road_2 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_2.txt',delimiter=',')
        road_3_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_3_0.txt',delimiter=',')
        road_3_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_3_1.txt',delimiter=',')
        road_4_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_4_0.txt',delimiter=',')
        road_4_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_4_1.txt',delimiter=',')
        road_5_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_5_0.txt',delimiter=',')
        road_5_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_5_1.txt',delimiter=',')

        road_6 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_6.txt',delimiter=',')
        road_7 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_7.txt',delimiter=',')
        road_8 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_8.txt',delimiter=',')
        road_9_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_9_0.txt',delimiter=',')
        road_9_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_9_1.txt',delimiter=',')
        road_10_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_10_0.txt',delimiter=',')
        road_10_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/new-loop/road_10_1.txt',delimiter=',')

        self._lanes.append(dense_polyline2d(road_0,1))
        self._lanes.append(dense_polyline2d(road_1,1))
        self._lanes.append(dense_polyline2d(road_2,1))
        self._lanes.append(dense_polyline2d(road_3_0,1))
        self._lanes.append(dense_polyline2d(road_4_0,1))
        self._lanes.append(dense_polyline2d(road_5_0,1))
        self._lanes.append(dense_polyline2d(road_6,1))
        self._lanes.append(dense_polyline2d(road_7,1))
        self._lanes.append(dense_polyline2d(road_8,1))
        self._lanes.append(dense_polyline2d(road_9_0,1))
        self._lanes.append(dense_polyline2d(road_10_0,1))

        self.road_3_1 = dense_polyline2d(road_3_1,1)
        self.road_4_1 = dense_polyline2d(road_4_1,1)
        self.road_5_1 = dense_polyline2d(road_5_1,1)
        self.road_9_1 = dense_polyline2d(road_9_1,1)
        self.road_10_1 = dense_polyline2d(road_10_1,1)

    def ego_road(self):

        dist_list = np.array([dist_from_point_to_polyline2d(
                self._ego_vehicle_x, self._ego_vehicle_y,
                lane) for lane in self._lanes])
        closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        return closest_lane

    def receive_new_pose(self, x, y):
        self._ego_vehicle_x_buffer = x
        self._ego_vehicle_y_buffer = y

    def update(self):
        self._ego_vehicle_x = self._ego_vehicle_x_buffer
        self._ego_vehicle_y = self._ego_vehicle_y_buffer
        self.ego_road_id = self.ego_road()

        if self.should_update_static_map():
            self._current_road_id = self.ego_road_id
            self._next_road_id = self._next_road_id_list[self.ego_road_id]
            self.update_static_map()
            return self.static_local_map

        return None

    def should_update_static_map(self):
        '''
        Determine whether map updating is needed.
        '''
        return True

        if self._current_road_id < 0:
            return True
        
        if self._current_road_id == self.ego_road_id:
            return False
        
        if self._next_road_id == self.ego_road_id:
            return True

        return False

    def init_static_map(self):
        '''
        Generate null static map
        '''
        init_static_map = Map()
        init_static_map.in_junction = False
        init_static_map.exit_lane_index.append(0)
        return init_static_map


    def update_static_map(self):
        ''' 
        Update information in the static map if current location changed dramatically
        '''
        map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y

        self.static_local_map = self.init_static_map()  ## Return this one
        self.update_lane_list(self.ego_road_id)
        print(" ###################  ",self.ego_road_id)
        self.update_next_road_list(self._next_road_id)

    def update_lane_list(self, road_id):
        '''
        Update lanes when a new road is encountered
        '''
        # map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y
        # Right is 0

        self.static_local_map.in_junction = False # lane change
        self.static_local_map.lanes.append(self.get_lane(self._lanes[road_id]))

        if road_id == 3:
            self.static_local_map.lanes.append(self.get_lane(self.road_3_1))
            self.static_local_map.exit_lane_index[0] = 0
        
        if road_id == 4:
            self.static_local_map.lanes.append(self.get_lane(self.road_4_1))
            self.static_local_map.exit_lane_index.append(0)

        if road_id == 5:
            self.static_local_map.lanes.append(self.get_lane(self.road_5_1))
            self.static_local_map.exit_lane_index.append(0)


        if road_id == 9:
            self.static_local_map.exit_lane_index[0] = 1
            self.static_local_map.lanes.append(self.get_lane(self.road_9_1))
        
        if road_id == 10:
            self.static_local_map.exit_lane_index[0] = 0
            self.static_local_map.lanes.append(self.get_lane(self.road_10_1))

        for i, lane in enumerate(self.static_local_map.lanes):
            if road_id == 0:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 1:
                lane.speed_limit = 15
                lane.traffic_light_pos.append(0)
            if road_id == 2:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 3:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 4:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 5:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 6:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 7:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 8:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 9:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 10:
                lane.speed_limit = 20
                lane.traffic_light_pos.append(0)

    def update_next_road_list(self, next_road_id):

        self.static_local_map.next_lanes.append(self.get_lane(self._lanes[next_road_id]))

        if next_road_id == 3:
            self.static_local_map.next_lanes.append(self.get_lane(self.road_3_1))
        if next_road_id == 4:
            self.static_local_map.next_lanes.append(self.get_lane(self.road_4_1))
        if next_road_id == 5:
            self.static_local_map.next_lanes.append(self.get_lane(self.road_5_1))
        if next_road_id == 9:
            self.static_local_map.next_lanes.append(self.get_lane(self.road_9_1))
        # if next_road_id == 10:
        #     self.static_local_map.next_lanes.append(self.get_lane(self.road_10_1))

    def get_lane(self, central_points): # outside

        lane_wrapped = Lane()
        for wp in central_points: # TODO Consider ego pose where is the start and end point.
            point = LanePoint()
            x, y = wp[0], wp[1]   # Point XY
            #x, y = wp[1], wp[0]
            point.position.x = x
            point.position.y = y
            lane_wrapped.central_path_points.append(point)

        return lane_wrapped


class TLMap(object):
    def __init__(self, offset_x=0, offset_y=0):
        self._ego_vehicle_x = None # location in meters
        self._ego_vehicle_y = None

        self._offset_x = offset_x # map center offset in meters
        self._offset_y = offset_y

        self._ego_vehicle_x_buffer = 0.0
        self._ego_vehicle_y_buffer = 0.0

        self._reference_lane_list = deque(maxlen=20000)
        self._lanes = []
        self.load_lanes() # load lanes
        self._next_road_id_list = [1, 0]

        self._current_road_id = -1
        self._next_road_id = -1

    def load_lanes(self):
        
        road_0 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/testing_loop/traffic_light/road_0.txt',delimiter=',')
        road_1 = np.loadtxt(os.environ.get('ZZZ_ROOT') + '/zzz/src/navigation/data/testing_loop/traffic_light/road_1.txt',delimiter=',')
        
        self._lanes.append(dense_polyline2d(road_0,1))
        self._lanes.append(dense_polyline2d(road_1,1))

    def ego_road(self):

        dist_list = np.array([dist_from_point_to_polyline2d(
                self._ego_vehicle_x, self._ego_vehicle_y,
                lane) for lane in self._lanes])
        closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        return closest_lane

    def receive_new_pose(self, x, y):
        self._ego_vehicle_x_buffer = x
        self._ego_vehicle_y_buffer = y

    def update(self):
        self._ego_vehicle_x = self._ego_vehicle_x_buffer
        self._ego_vehicle_y = self._ego_vehicle_y_buffer
        self.ego_road_id = self.ego_road()

        if self.should_update_static_map():
            self._current_road_id = self.ego_road_id
            self._next_road_id = self._next_road_id_list[self.ego_road_id]
            self.update_static_map()
            return self.static_local_map

        return None

    def should_update_static_map(self):
        '''
        Determine whether map updating is needed.
        '''
        return True

        if self._current_road_id < 0:
            return True
        
        if self._current_road_id == self.ego_road_id:
            return False
        
        if self._next_road_id == self.ego_road_id:
            return True

        return False

    def init_static_map(self):
        '''
        Generate null static map
        '''
        init_static_map = Map()
        init_static_map.in_junction = False
        init_static_map.exit_lane_index.append(0)
        return init_static_map


    def update_static_map(self):
        ''' 
        Update information in the static map if current location changed dramatically
        '''
        map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y

        self.static_local_map = self.init_static_map()  ## Return this one
        self.update_lane_list(self.ego_road_id)
        print(" ###################  ",self.ego_road_id)
        self.update_next_road_list(self._next_road_id)

    def update_lane_list(self, road_id):
        '''
        Update lanes when a new road is encountered
        '''
        # map_x, map_y = self._ego_vehicle_x, self._ego_vehicle_y
        # Right is 0

        self.static_local_map.in_junction = False # lane change
        self.static_local_map.lanes.append(self.get_lane(self._lanes[road_id]))

        for i, lane in enumerate(self.static_local_map.lanes):
            if road_id == 0:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 1:
                lane.speed_limit = 15
                lane.traffic_light_pos.append(0)
            if road_id == 2:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 3:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 4:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 5:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 6:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 7:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 8:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 9:
                lane.speed_limit = 30
                lane.traffic_light_pos.append(0)
            if road_id == 10:
                lane.speed_limit = 20
                lane.traffic_light_pos.append(0)

    def update_next_road_list(self, next_road_id):

        self.static_local_map.next_lanes.append(self.get_lane(self._lanes[next_road_id]))

    def get_lane(self, central_points): # outside

        lane_wrapped = Lane()
        for wp in central_points: # TODO Consider ego pose where is the start and end point.
            point = LanePoint()
            x, y = wp[0], wp[1]   # Point XY
            #x, y = wp[1], wp[0]
            point.position.x = x
            point.position.y = y
            lane_wrapped.central_path_points.append(point)

        return lane_wrapped

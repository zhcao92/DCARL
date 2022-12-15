#!/usr/bin/env python

import rospy
import numpy as np
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from zzz_cognition_msgs.msg import MapState
from zzz_driver_msgs.utils import get_speed, get_yaw

from predict import predict

class MainDecision(object):
    def __init__(self, trajectory_planner=None):
        self._dynamic_map_buffer = None
        self._trajectory_planner = trajectory_planner

    def receive_dynamic_map(self, dynamic_map):
        self._dynamic_map_buffer = dynamic_map

    def update_trajectory(self):
        
        # This function generate trajectory

        close_to_junction = 40

        # Update_dynamic_local_map
        if self._dynamic_map_buffer is None:
            return None
        dynamic_map = self._dynamic_map_buffer

        if dynamic_map.model == dynamic_map.MODEL_MULTILANE_MAP and dynamic_map.mmap.distance_to_junction > close_to_junction:
            self._trajectory_planner.clear_buff(dynamic_map)
            return None
        elif dynamic_map.model == dynamic_map.MODEL_MULTILANE_MAP:
            msg = self._trajectory_planner.build_frenet_path(dynamic_map)
            return None
        else:
            return self._trajectory_planner.trajectory_update(dynamic_map)

        


    

    

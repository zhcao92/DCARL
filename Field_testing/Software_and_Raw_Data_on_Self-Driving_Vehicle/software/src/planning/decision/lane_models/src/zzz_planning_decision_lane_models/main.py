#!/usr/bin/env python

import rospy
import numpy as np
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from zzz_planning_msgs.msg import DecisionTrajectory
from threading import Lock
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from zzz_cognition_msgs.msg import MapState
from zzz_driver_msgs.utils import get_speed, get_yaw
from zzz_planning_decision_lane_models.local_trajectory import PolylineTrajectory, Werling_planner # TODO(Temps): Should seperate into continous models

# Make lat lon model as parameter

class MainDecision(object):
    def __init__(self, lon_decision=None, lat_decision=None, local_trajectory=None):
        self._dynamic_map_buffer = None
        self._static_map_buffer = None

        self._longitudinal_model_instance = lon_decision
        self._lateral_model_instance = lat_decision
        self._local_trajectory_instance = Werling_planner()
        # self._local_trajectory_instance_for_ref = PolylineTrajectory() # TODO(Temps): Should seperate into continous models

        self._dynamic_map_lock = Lock()

    # receive_dynamic_map running in Subscriber CallBack Thread.
    def receive_dynamic_map(self, dynamic_map):
        self._dynamic_map_buffer = dynamic_map

    # update running in main node thread loop
    def update(self):
        '''
        This function generate trajectory
        '''
        # update_dynamic_local_map
        if self._dynamic_map_buffer is None:
            return None
        else:
            dynamic_map = self._dynamic_map_buffer

        # close_to_lane = 1

        if dynamic_map.model == dynamic_map.MODEL_JUNCTION_MAP: 
            self._local_trajectory_instance.clean_frenet_lane()
            return None
        

        trajectory = None
        changing_lane_index, desired_speed = self._lateral_model_instance.lateral_decision(dynamic_map)
        
        ego_speed = get_speed(dynamic_map.ego_state)

        rospy.logdebug("target_lane_index = %d, target_speed = %f km/h, current_speed: %f km/h", changing_lane_index, desired_speed*3.6, ego_speed*3.6)
        
        trajectory, local_desired_speed = self._local_trajectory_instance.get_trajectory(dynamic_map, changing_lane_index, desired_speed)

        if local_desired_speed <= 0.1/3.6 and ego_speed > 3/3.6: # TODO: clean this
            local_desired_speed = 0.1/3.6

        msg = DecisionTrajectory()
        msg.trajectory = self.convert_ndarray_to_pathmsg(trajectory) # TODO: move to library
        msg.desired_speed = local_desired_speed # TODO: Multi-resolution Planning

        return msg

    def convert_ndarray_to_pathmsg(self, path): 
        msg = Path()
        for wp in path:
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            msg.poses.append(pose)
        msg.header.frame_id = "map"

        return msg

    

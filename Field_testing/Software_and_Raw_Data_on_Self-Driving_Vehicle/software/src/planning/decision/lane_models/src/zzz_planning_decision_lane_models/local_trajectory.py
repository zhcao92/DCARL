import rospy
import numpy as np
import math
# from cvxopt import solvers, matrix
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from zzz_cognition_msgs.msg import MapState
from zzz_driver_msgs.utils import get_speed, get_yaw
from lane_werling_planner import Werling
from zzz_navigation_msgs.msg import LanePoint



class PolylineTrajectory(object):
    def get_trajectory(self, dynamic_map, target_lane_index, desired_speed,
                resolution=0.5, time_ahead=5, distance_ahead=10, rectify_thres=2,
                lc_dt = 1.5, lc_v = 2.67):
        # TODO: get smooth spline (write another module to generate spline)
        ego_x = dynamic_map.ego_state.pose.pose.position.x
        ego_y = dynamic_map.ego_state.pose.pose.position.y
        if target_lane_index == -1:
            target_lane = dynamic_map.jmap.reference_path
        else:
            target_lane = dynamic_map.mmap.lanes[int(target_lane_index)]

        central_path = self.convert_path_to_ndarray(target_lane.map_lane.central_path_points)

        # if ego vehicle is on the target path
        # works for lane change, lane follow and reference path follow
        dense_centrol_path = dense_polyline2d(central_path, resolution)
        nearest_dis, nearest_idx, _ = dist_from_point_to_polyline2d(ego_x, ego_y, dense_centrol_path)
        nearest_dis = abs(nearest_dis)

        if nearest_dis > rectify_thres:
            if dynamic_map.model == MapState.MODEL_MULTILANE_MAP and target_lane_index != -1:
                rectify_dt = abs(dynamic_map.mmap.ego_lane_index - target_lane_index)*lc_dt
            else:
                rectify_dt = nearest_dis/lc_v
            return self.generate_smoothen_lane_change_trajectory(dynamic_map, target_lane, rectify_dt, desired_speed)
        else:
            front_path = dense_centrol_path[nearest_idx:]
            dis_to_ego = np.cumsum(np.linalg.norm(np.diff(front_path, axis=0), axis = 1))
            trajectory = front_path[:np.searchsorted(dis_to_ego, desired_speed*time_ahead+distance_ahead)-1]
            return trajectory

    # TODO(zyxin): Add these to zzz_navigation_msgs.utils
    def convert_path_to_ndarray(self, path):
        point_list = [(point.position.x, point.position.y) for point in path]
        return np.array(point_list)

    def generate_smoothen_lane_change_trajectory(self, dynamic_map, target_lane,
        rectify_dt, desired_speed, lc_dt = 1.5, rectify_min_d = 6, resolution=0.5, time_ahead=5, distance_ahead=10):

        target_lane_center_path = self.convert_path_to_ndarray(target_lane.map_lane.central_path_points)

        ego_x = dynamic_map.ego_state.pose.pose.position.x
        ego_y = dynamic_map.ego_state.pose.pose.position.y

        # Calculate the longitudinal distance for lane Change
        # Considering if the ego_vehicle is in a lane Change
        lc_dis = max(rectify_dt*desired_speed,rectify_min_d)

        dense_target_centrol_path = dense_polyline2d(target_lane_center_path, resolution)
        _, nearest_idx, _ = dist_from_point_to_polyline2d(ego_x, ego_y, dense_target_centrol_path)
        front_path = dense_target_centrol_path[nearest_idx:]
        dis_to_ego = np.cumsum(np.linalg.norm(np.diff(front_path, axis=0), axis = 1))

        start_point = np.array([ego_x,ego_y])
        end_point = front_path[np.searchsorted(dis_to_ego, lc_dis)]

        # calculate start direction and end direction for control

        ego_direction = get_yaw(dynamic_map.ego_state)
        _, nearest_end_idx, _ = dist_from_point_to_polyline2d(end_point[0], end_point[1], target_lane_center_path)
        end_point_direction = target_lane.map_lane.central_path_points[nearest_end_idx].tangent

        start_tangent = np.array([np.cos(ego_direction),np.sin(ego_direction)])
        end_tangent = np.array([np.cos(end_point_direction),np.sin(end_point_direction)])

        lc_path = self.cubic_hermite_spline(p0 = start_point, p1 = end_point,
                                            m0 = start_tangent, m1 = end_tangent)

        # get ahead Path
        ahead_dis = desired_speed*time_ahead+distance_ahead
        path_after_lc = front_path[np.searchsorted(dis_to_ego, lc_dis):np.searchsorted(dis_to_ego, ahead_dis)-1]

        # replace lane change path into ahead path
        smoothen_lc_path = np.concatenate((lc_path,path_after_lc),axis = 0)
        return smoothen_lc_path

    def cubic_hermite_spline(self, p0, p1, m0, m1, resolution = 20):
        """
        Generate cubic hermit spline
        p0: start point np.array(2)
        p1: end point np.array(2)
        m0: start tangent np.array(2)
        m1: end tangent np.array(2)
        return path from start point to end point
        """

        t = np.linspace(0,1,num = resolution)
        h00 = (2*t*t*t-3*t*t+1).reshape(len(t),1) #p0
        h10 = (t*t*t-2*t*t+t).reshape(len(t),1) #m0
        h01 = (-2*t*t*t+3*t*t).reshape(len(t),1) #p1
        h11 = (t*t*t-t*t).reshape(len(t),1) #m1

        p0 = p0.reshape(1,2)
        p1 = p1.reshape(1,2)
        m0 = m0.reshape(1,2)
        m1 = m1.reshape(1,2)

        return np.matmul(h00,p0) + np.matmul(h10,m0) + np.matmul(h01,p1) + np.matmul(h11,m1)


class Werling_planner(object):
    def __init__(self):
        self.last_target_lane_index = -10
        self.lanes = []


    def build_frenet_lane(self,dynamic_map):
        if len(self.lanes) > 0:
            return

        lane_num = len(dynamic_map.mmap.lanes)
        for lane_idx,lane in enumerate(dynamic_map.mmap.lanes):
            central_path = lane.map_lane.central_path_points
            extend_centrol_path = self.extend_path(central_path)
            self.lanes.append(Werling(extend_centrol_path,lane_idx,lane_num))

    def clean_frenet_lane(self):
        self.lanes = []
        self.last_target_lane_index = -1

    def get_trajectory(self, dynamic_map, target_lane_index, desired_speed,
                resolution=2.0, time_ahead=5, distance_ahead=10, rectify_thres=2,
                lc_dt = 1.5, lc_v = 2.67):
        # TODO: get smooth spline (write another module to generate spline)
        # ego_x = dynamic_map.ego_state.pose.pose.position.x
        # ego_y = dynamic_map.ego_state.pose.pose.position.y

        ego_lane_idx = int(round(dynamic_map.mmap.ego_lane_index))

        # if len(self.lanes) <= 0:
        if self.last_target_lane_index < 0:
            self.build_frenet_lane(dynamic_map)

        if target_lane_index != self.last_target_lane_index:
            for werlinglane in self.lanes:
                werlinglane.lane_change_clean_buff()

            self.last_target_lane_index = target_lane_index
            
        return self.lanes[int(target_lane_index)].trajectory_update(dynamic_map, desired_speed, ego_lane_idx)

    def extend_path(self, path):
        
        dx = path[1].position.x - path[0].position.x
        dy = path[1].position.y - path[0].position.y
        
        for i in range(1,30):
            point = LanePoint()
            point.position.x = path[0].position.x - dx
            point.position.y = path[0].position.y - dy
            path.insert(0, point)

        return path

    def get_rviz_info(self):
        
        if self.last_target_lane_index < 0:
            return None, None, None
        
        all_trajectory = self.lanes[self.last_target_lane_index].rivz_element.candidates_trajectory
        obs_paths = self.lanes[self.last_target_lane_index].rivz_element.prediciton_trajectory
        collision_circle = self.lanes[self.last_target_lane_index].rivz_element.collision_circle

        return all_trajectory, obs_paths, collision_circle

    # TODO(zyxin): Add these to zzz_navigation_msgs.utils
    def convert_path_to_ndarray(self, path):
        point_list = [(point.position.x, point.position.y) for point in path]
        return np.array(point_list)

    
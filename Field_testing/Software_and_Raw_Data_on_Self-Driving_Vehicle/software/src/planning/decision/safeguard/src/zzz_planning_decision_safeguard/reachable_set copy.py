
import rospy
import numpy as np
import math

from zzz_perception_msgs.msg import ObjectClass
from zzz_driver_msgs.utils import get_speed, get_yaw
from zzz_cognition_msgs.msg import MapState
from intervaltree import Interval, IntervalTree
from zzz_common.geometry import dist_from_point_to_polyline2d

class ReachableSet(object):

    def __init__(self):
        self.dynamic_map = None
        self.vehicle_list = []
        self.pedestrian_list = []
        self.collision_points = []

    def update_dynamic_map(self, dynamic_map):
        self.dynamic_map = dynamic_map
        self.vehicle_list = dynamic_map.jmap.obstacles
        # TODO(zyxin): Wait for Carla fix this issue
        # self.vehicle_list = [obj for obj in dynamic_map.jmap.obstacles if obj.cls.classid & ObjectClass.LEVEL_MASK_0 == ObjectClass.VEHICLE]
        # self.pedestrian_list = [obj for obj in dynamic_map.jmap.obstacles if obj.cls.classid & ObjectClass.LEVEL_MASK_0 == ObjectClass.HUMAN]
    
    def check_trajectory(self, decision_trajectory, desired_speed):

        decision_trajectory_array = self.convert_trajectory_to_ndarray(decision_trajectory)
        if len(decision_trajectory_array) == 0:
            return False, desired_speed
        self.collision_points = []
        ego_idx = self.get_ego_idx(decision_trajectory_array)
        nearest_idx = len(decision_trajectory_array)-1
        nearest_obstacle = None
        
        # FIXME(zyxin): should be in prediction part
        if self.dynamic_map.model == MapState.MODEL_JUNCTION_MAP or self.dynamic_map is None: 
            for vehicle in self.vehicle_list:
                if vehicle.lane_index > -1:
                    continue
                pred_trajectory = self.pred_trajectory(vehicle)
                collision_idx, collision_time = self.intersection_between_trajectories(decision_trajectory_array,
                                                                                            pred_trajectory,vehicle)
                #TODO:ignore a near distance
                if collision_time < float("inf"):
                    self.collision_points.append((collision_idx,collision_time))

        # TODO: adjust for pedestrain
        # for pedestrian in self.pedestrian_list:
        #     pred_trajectory = self.pred_trajectory(pedestrian)
        #     collision_idx, collision_time = self.intersection_between_trajectories(decision_trajectory_array,pred_trajectory,vehicle)
        #     if collision_idx > ego_idx and collision_idx < nearest_idx:
        #         nearest_idx = collision_idx
        #         nearest_obstacle = pedestrian

        safeguard_speed = self.get_safeguard_speed(ego_idx,decision_trajectory_array,desired_speed)

        if safeguard_speed < desired_speed:
            return True, safeguard_speed
        else:
            return False, desired_speed

    def pred_trajectory(self, obstacle, pred_t=4, resolution=0.1):
        # Assuming constant speed
        loc = np.array([obstacle.state.pose.pose.position.x, obstacle.state.pose.pose.position.y])
        speed = get_speed(obstacle.state)
        yaw = get_yaw(obstacle.state)
        speed_direction = np.array([np.cos(yaw), np.sin(yaw)])
        t_space = np.linspace(0, pred_t, pred_t/resolution)

        pred_x = loc[0] + t_space*speed*speed_direction[0]
        pred_y = loc[1] + t_space*speed*speed_direction[1]
        pred_trajectory = np.array([pred_x, pred_y]).T

        return pred_trajectory

    def intersection_between_trajectories(self, decision_trajectory, pred_trajectory, vehicle, 
                                 collision_thres=4, stop_thres=3, unavoidable_distance = 4):
        '''
        If two trajectory have interection, return the distance
        TODO: Implement this method into a standalone library
        '''
        
        nearest_idx = len(decision_trajectory)-1
        collision_time = float("inf")
        surrounding_vehicle_speed = get_speed(vehicle.state)

        # the object stops
        # if np.linalg.norm(pred_trajectory[0] - pred_trajectory[-1]) < stop_thres:
        if surrounding_vehicle_speed < stop_thres:
            return nearest_idx, collision_time

        ego_pose = self.dynamic_map.ego_state.pose.pose
        ego_loc = np.array([ego_pose.position.x,ego_pose.position.y])

        for wp in pred_trajectory:
            dist = np.linalg.norm(decision_trajectory - wp,axis=1)
            min_d = np.min(dist)
            distance_to_ego_vehicle = np.linalg.norm(wp - ego_loc)
            if distance_to_ego_vehicle < unavoidable_distance:
                break

            if min_d < collision_thres:
                nearest_idx = np.argmin(dist)
                surrounding_vehicle_distance_before_collision = dist_from_point_to_polyline2d(wp[0],wp[1],pred_trajectory)[2]
                collision_time = surrounding_vehicle_distance_before_collision/surrounding_vehicle_speed
                # rospy.logdebug("considering vehicle:%d, dis_to_collision:%d, col_time:%.2f, speed:%.2f col_index:%d(%d)",vehicle.uid,
                #                                                     surrounding_vehicle_distance_before_collision,
                #                                                     collision_time,surrounding_vehicle_speed,
                #                                                     nearest_idx,len(decision_trajectory))
                break
            
        return nearest_idx,collision_time
        

    def convert_trajectory_to_ndarray(self, trajectory):

        trajectory_array = [(pose.pose.position.x, pose.pose.position.y) for pose in trajectory.poses]
        return np.array(trajectory_array)


    def get_ego_idx(self, decision_trajectory):
        ego_pose = self.dynamic_map.ego_state.pose.pose
        ego_loc = np.array([ego_pose.position.x,ego_pose.position.y])
        dist = np.linalg.norm(decision_trajectory-ego_loc, axis=1)
        ego_idx = np.argmin(dist)
        return ego_idx

    def get_safeguard_speed(self, ego_idx, decision_trajectory, desired_speed, pred_t = 4, safe_dis_range = 7):
        
        danger_speed_range = IntervalTree()
        for collision_idx, collision_time in self.collision_points:
            if collision_idx>len(decision_trajectory)-1:
                continue
            dis_sum = np.cumsum(np.linalg.norm(np.diff(decision_trajectory, axis=0), axis=1))
            dis = dis_sum[collision_idx-1] - dis_sum[ego_idx]

            speed_min = max(0,((dis-safe_dis_range)/collision_time))
            speed_max = min(desired_speed+0.01,(dis+safe_dis_range)/collision_time)
            if speed_min >= speed_max:
                continue
            # rospy.logdebug("col_idx:%d,col_time:%.2f,ocp_min:%.2f,ocp_max:%.2f,desired_speed:%.2f)",
            #                                                                 collision_idx,collision_time,
            #                                                                 speed_min,speed_max,desired_speed)
            danger_speed_range[speed_min:speed_max]=(speed_min,speed_max)


        if len(danger_speed_range[desired_speed]) == 0:
            return desired_speed
        else:
            danger_speed_range.merge_overlaps()
            speed = sorted(danger_speed_range)[-1].begin
            return speed
        

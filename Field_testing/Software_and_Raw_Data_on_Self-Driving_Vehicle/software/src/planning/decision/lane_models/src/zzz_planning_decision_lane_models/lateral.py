
import rospy
import numpy as np
import math

from zzz_driver_msgs.utils import get_speed
from zzz_cognition_msgs.msg import MapState, RoadObstacle

class LaneUtility(object):

    def __init__(self, longitudinal_model):
        self.longitudinal_model_instance = longitudinal_model
        self.dynamic_map = None

    def lateral_decision(self, dynamic_map, close_to_junction = 10):

        self.longitudinal_model_instance.update_dynamic_map(dynamic_map)
        self.dynamic_map = dynamic_map
        # return -1, self.longitudinal_model_instance.longitudinal_speed(-1)#FIXME(ksj)
        rospy.logdebug("map model is %d", dynamic_map.model)
       
        target_index = self.generate_lane_change_index()
        target_speed = self.longitudinal_model_instance.longitudinal_speed(target_index,traffic_light = True)

        tail_speed = self.tail_speed(dynamic_map.mmap.distance_to_junction)

        if tail_speed < target_speed:
            target_speed = tail_speed

        return target_index, target_speed

    def generate_lane_change_index(self, change_lane_thres = 0.5):

        ego_lane_index = int(round(self.dynamic_map.mmap.ego_lane_index))
        current_lane_utility = self.lane_utility(ego_lane_index) + change_lane_thres

        if not self.lane_change_safe(ego_lane_index, ego_lane_index + 1):
            left_lane_utility = -1
        else:
            left_lane_utility = self.lane_utility(ego_lane_index + 1)

        if not self.lane_change_safe(ego_lane_index, ego_lane_index - 1):
            right_lane_utility = -1
        else:
            right_lane_utility = self.lane_utility(ego_lane_index - 1)

        # TODO: target lane = -1?
        rospy.logdebug("left_utility = %f, ego_utility = %f, right_utility = %f",
            left_lane_utility, current_lane_utility, right_lane_utility)

        if right_lane_utility > current_lane_utility and right_lane_utility >= left_lane_utility:
            return ego_lane_index - 1

        if left_lane_utility > current_lane_utility and left_lane_utility > right_lane_utility:
            return ego_lane_index + 1

        return ego_lane_index

    def lane_utility(self, lane_index):

        available_speed = self.longitudinal_model_instance.longitudinal_speed(lane_index)
        exit_lane_index = self.dynamic_map.mmap.target_lane_index
        distance_to_end = self.dynamic_map.mmap.distance_to_junction
        utility = available_speed*1.5 + 1/(abs(exit_lane_index - lane_index)+1)*max(0,(200-distance_to_end))*0.1
        return utility

    def lane_change_safe(self, ego_lane_index, target_index):

        if target_index < 0 or target_index > len(self.dynamic_map.mmap.lanes)-1:
            return False

        front_safe = False
        rear_safe = False
        front_vehicle = None
        rear_vehicle = None
        ego_vehicle_location = np.array([self.dynamic_map.ego_state.pose.pose.position.x,
                                         self.dynamic_map.ego_state.pose.pose.position.y])

        target_lane = self.dynamic_map.mmap.lanes[target_index]

        if len(target_lane.front_vehicles) > 0:
            front_vehicle = target_lane.front_vehicles[0]
        
        if len(target_lane.rear_vehicles) > 0:
            rear_vehicle = target_lane.rear_vehicles[0]

        # for lane in self.dynamic_map.mmap.lanes:
        #     if lane.map_lane.index == target_index:
        #         if len(lane.front_vehicles) > 0:
        #             front_vehicle = lane.front_vehicles[0]
        #         if len(lane.rear_vehicles) > 0:
        #             rear_vehicle = lane.rear_vehicles[0]
        #         break

        ego_v = get_speed(self.dynamic_map.ego_state)

        if front_vehicle is None:
            d_front = -1
            front_safe = True
            behavior_front = RoadObstacle.BEHAVIOR_UNKNOWN
        else:
            front_vehicle_location = np.array([
                front_vehicle.state.pose.pose.position.x,
                front_vehicle.state.pose.pose.position.y
            ])
            behavior_front = front_vehicle.behavior
            # TODO: Change to real distance in lane
            d_front = np.linalg.norm(front_vehicle_location - ego_vehicle_location)
            front_v = get_speed(front_vehicle.state)
            if d_front > max(10 + 3*(ego_v-front_v), 20):
                front_safe = True
        

        if rear_vehicle is None:
            rear_safe = True
            d_rear = -1
            behavior_rear = RoadObstacle.BEHAVIOR_UNKNOWN
        else:
            rear_vehicle_location = np.array([
                rear_vehicle.state.pose.pose.position.x,
                rear_vehicle.state.pose.pose.position.y
            ])
            behavior_rear = rear_vehicle.behavior
            d_rear = np.linalg.norm(rear_vehicle_location - ego_vehicle_location)
            rear_v = get_speed(rear_vehicle.state)
            if d_rear > max(10 + 3*(rear_v-ego_v), 20):
                rear_safe = True

        rospy.logdebug("ego_lane = %d, target_lane = %d, front_d = %f(%d), rear_d = %f(%d)",
                                ego_lane_index, target_index, d_front, 
                                behavior_front, d_rear, behavior_rear)
                                
        if front_safe and rear_safe:
            return True
            
        return False

    def tail_speed(self, d):
        '''
        Calculate the speed when close to the tail
        TODO(zhcao): should merge in Control model
        '''
        if d <= 0:
            return 0.0

        dec = 0.4
        available_speed = math.sqrt(2*dec*d) # m/s
        ego_v = get_speed(self.dynamic_map.ego_state)
        if available_speed > ego_v:
            return 10000
        
        dt = 0.4
        vehicle_dec = (ego_v - available_speed)*5
        
        return ego_v - vehicle_dec*dt
 
class MOBIL(object):
    pass

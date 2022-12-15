import numpy as np
import math
from actions import LaneAction


class LanePlanner(object):
    """
    Lane Level Planner, including target lane and speed
    """
    
    def __init__(self):
        pass

    def run_step(self, dynamic_map):

        self.longitudinal_model_instance = IDM(dynamic_map)
        self.dynamic_map = dynamic_map

        target_index = self.generate_lane_change_index()
        target_speed = self.longitudinal_model_instance.longitudinal_speed(target_index)
        # print("[IDM] : target line index",self.dynamic_map.ego_vehicle.lane_idx,target_index,len(self.dynamic_map.lanes))

        return LaneAction(target_index, target_speed)

    def generate_lane_change_index(self, change_lane_thres=0.5):

        ego_lane_index = int(round(self.dynamic_map.ego_vehicle.lane_idx))
        
        current_lane_utility = self.lane_utility(ego_lane_index) + change_lane_thres

        if not self.lane_change_safe(ego_lane_index, ego_lane_index + 1):
            left_lane_utility = -1
        else:
            left_lane_utility = self.lane_utility(ego_lane_index + 1)

        if not self.lane_change_safe(ego_lane_index, ego_lane_index - 1):
            right_lane_utility = -1
        else:
            right_lane_utility = self.lane_utility(ego_lane_index - 1)

        if right_lane_utility > current_lane_utility and right_lane_utility >= left_lane_utility:
            return ego_lane_index - 1

        if left_lane_utility > current_lane_utility and left_lane_utility > right_lane_utility:
            return ego_lane_index + 1

        return ego_lane_index

    def lane_utility(self, lane_index):
        available_speed = self.longitudinal_model_instance.longitudinal_speed(lane_index)
        utility = available_speed
        return utility

    def lane_change_safe(self, ego_lane_index, target_index):

        if target_index < 0 or target_index > len(self.dynamic_map.lanes)-1:
            return False

        front_safe = False
        rear_safe = False
        front_vehicle = None
        rear_vehicle = None
        ego_vehicle_location = np.array([self.dynamic_map.ego_vehicle.x,
                                         self.dynamic_map.ego_vehicle.y])

        target_lane = self.dynamic_map.lanes[target_index]

        if len(target_lane.front_vehicles) > 0:
            front_vehicle = target_lane.front_vehicles[0]
        
        if len(target_lane.rear_vehicles) > 0:
            rear_vehicle = target_lane.rear_vehicles[0]

        ego_v = self.dynamic_map.ego_vehicle.v

        if front_vehicle is None:
            d_front = -1
            front_safe = True
        else:
            front_vehicle_location = np.array([
                        front_vehicle.x,front_vehicle.y])
            d_front = np.linalg.norm(front_vehicle_location - ego_vehicle_location)
            front_v = front_vehicle.v

            if d_front > max(10 + 3*(ego_v-front_v), 5):
                front_safe = True

        if rear_vehicle is None:
            rear_safe = True
            d_rear = -1
        else:
            rear_vehicle_location = np.array([rear_vehicle.x,rear_vehicle.y])
            d_rear = np.linalg.norm(rear_vehicle_location - ego_vehicle_location)
            rear_v = rear_vehicle.v

            if d_rear > max(10 + 3*(rear_v-ego_v), 5):
                rear_safe = True
                                
        if front_safe and rear_safe:
            return True
            
        return False


class IDM(object):

    def __init__(self, dynamic_map):

        self.T = 1.6
        self.g0 = 7
        self.a = 2.73 
        self.b = 1.65
        self.delta = 4
        self.decision_dt = 0.75
        self.dynamic_map = dynamic_map

    def longitudinal_speed(self, target_lane_index):

        target_lane = self.dynamic_map.lanes[target_lane_index]
        idm_speed = self.IDM_speed_in_lane(target_lane)

        return idm_speed

    def IDM_speed_in_lane(self, lane):

        ego_vehicle_location = np.array([self.dynamic_map.ego_vehicle.x, 
                                         self.dynamic_map.ego_vehicle.y])

        v = self.dynamic_map.ego_vehicle.v
        
        v0 = lane.speed_limit

        if v0 <= 0.0:
            return 0.0
        
        if v < 5:
            a = self.a + (5 - v) / 5*2
        else:
            a = self.a
        
        b = self.b
        g0 = self.g0
        T = self.T
        delta = self.delta
        # for vehicles in lane.front_vehicles:
            # print("[IDM] : vehicles dist:",vehicles.x, vehicles.y)


        if len(lane.front_vehicles) > 0:
            f_v_location = np.array([
                lane.front_vehicles[0].x,
                lane.front_vehicles[0].y])

            v_f = lane.front_vehicles[0].v
            # print("[IDM] : lane.front_vehicles:",len(lane.front_vehicles),lane.front_vehicles[0].x,lane.front_vehicles[0].y,v_f)

            dv = v - v_f
            g = np.linalg.norm(f_v_location - ego_vehicle_location)

            if g <= 0:
                return 0

            g1 = g0 + T * v + v * dv / (2 * np.sqrt(a * b))
            acc = a * (1 - pow(v/v0, delta) - (g1/g) * ((g1/g)))

        else:
            acc = a * (1 - pow(v/v0, delta))
        return max(0, v + acc*self.decision_dt)

        return False

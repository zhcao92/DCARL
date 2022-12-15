
import numpy as np
import rospy

from zzz_navigation_msgs.msg import Lane
from zzz_driver_msgs.utils import get_speed
from zzz_cognition_msgs.msg import RoadObstacle

class IDM(object):

    def __init__(self):
        self.T = 3.6
        self.g0 = 7 + 12
        self.a = 2.73 
        self.b = 1.65 + 5
        self.delta = 4
        self.decision_dt = 0.2
        self.dynamic_map = None
    
    def update_dynamic_map(self, dynamic_map):
        self.dynamic_map = dynamic_map

    def longitudinal_speed(self, target_lane_index, traffic_light = False):

        target_lane = None

        if target_lane_index > len(self.dynamic_map.mmap.lanes)-1:
            rospy.logwarn("cannot find neighbor lane, lane_index: %d", target_lane_index)
            return 0 

        target_lane = self.dynamic_map.mmap.lanes[target_lane_index]

        idm_speed = self.IDM_speed_in_lane(target_lane)
        # Response to front vehicle in left lane
        if target_lane_index < len(self.dynamic_map.mmap.lanes)-1:
            left_lane = None
            for lane in self.dynamic_map.mmap.lanes:
                if lane.map_lane.index == target_lane_index+1:
                    left_lane = lane
                    break
            if left_lane is not None and self.neighbor_vehicle_is_cutting_in(left_lane,target_lane):
                rospy.logwarn("response to left front vehicle(%d)",target_lane_index+1)
                left_idm_speed = self.IDM_speed_in_lane(left_lane)
                idm_speed = min(idm_speed,left_idm_speed)

        # Response to front vehicle in right lane
        if target_lane_index > 0:
            for lane in self.dynamic_map.mmap.lanes:
                right_lane = None
                if lane.map_lane.index == target_lane_index-1:
                    right_lane = lane
                    break
            if right_lane is not None and self.neighbor_vehicle_is_cutting_in(right_lane,target_lane):
                rospy.logwarn("response to right front vehicle(%d)",target_lane_index-1)
                right_idm_speed = self.IDM_speed_in_lane(right_lane)
                idm_speed = min(idm_speed,right_idm_speed)

        traffic_light_speed = float("inf")
        if traffic_light:
            traffic_light_speed = self.traffic_light_speed(target_lane)
        return min(idm_speed, traffic_light_speed)


    def IDM_speed_in_lane(self, lane):

        ego_vehicle_location = np.array([self.dynamic_map.ego_state.pose.pose.position.x,
                                         self.dynamic_map.ego_state.pose.pose.position.y])

        v = get_speed(self.dynamic_map.ego_state)
        v0 = lane.map_lane.speed_limit/3.6
        if v < 5:
            a = self.a + (5 - v)/5*2
        else:
            a = self.a
        
        b = self.b
        g0 = self.g0
        T = self.T
        delta = self.delta
        
        if len(lane.front_vehicles) > 0:
            f_v_location = np.array([
                lane.front_vehicles[0].state.pose.pose.position.x,
                lane.front_vehicles[0].state.pose.pose.position.y
            ])
            v_f = get_speed(lane.front_vehicles[0].state) # TODO: get mmap vx and vy, the translator part in nearest locator
            dv = v - v_f
            g = np.linalg.norm(f_v_location - ego_vehicle_location)
            g1 = g0 + T*v + v*dv/(2*np.sqrt(a*b))
        else:
            dv = 0
            g = 50
            g1 = 0

        if g == 0 or v0 == 0:
            rospy.logerr("Front vehicle position: (%.3f, %.3f), ego vehicle position: (%.3f, %.3f)", f_v_location[0], f_v_location[1], ego_vehicle_location[0], ego_vehicle_location[1])
        acc = a*(1 - pow(v/v0, delta) - (g1/g)*((g1/g)))

        return max(0, v + acc*self.decision_dt)


    def traffic_light_speed(self, lane):

        ego_vehicle_speed = get_speed(self.dynamic_map.ego_state)
        if lane.map_lane.stop_state == Lane.STOP_STATE_THRU:
            return float("inf")
        else:
            d = lane.stop_distance
            if d < 10 + ego_vehicle_speed*ego_vehicle_speed/2/2:
                return 0 

        return float("inf")


    def neighbor_vehicle_is_cutting_in(self,neighbor_lane,ego_lane):
        if len(neighbor_lane.front_vehicles) == 0:
            return False
        
        if neighbor_lane.front_vehicles[0].behavior is not RoadObstacle.BEHAVIOR_MOVING_LEFT \
                            and neighbor_lane.front_vehicles[0].behavior is not RoadObstacle.BEHAVIOR_MOVING_RIGHT:
            return False
        
        mmap_y = neighbor_lane.front_vehicles[0].ffstate.d

        ego_idx = int(round(ego_lane.map_lane.index))
        neighbor_idx = int(round(neighbor_lane.map_lane.index))

        if ((neighbor_idx-mmap_y)*(ego_idx-mmap_y)) < 0:
            # rospy.logdebug("cut in judgement: ego_lane:%d, neighbor_lane:%d, mmap_y:%f",ego_idx,neighbor_idx,mmap_y)
            return True

        return False



class Gipps(object):
    # TODO(zhcao): Implement Gipps model
    pass

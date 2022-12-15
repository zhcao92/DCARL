import numpy as np
import math
# import sumolib
import threading
import time


class Position():
    def __init__(self):
        self.x = 0
        self.y = 0

class Lanepoint():
    def __init__(self):
        self.position = Position()


class Lane():
    def __init__(self):
        self.speed_limit = 30/3.6 #m/s
        self.lane_index = None
        self.central_path = []
        self.central_path_array = []
        self.front_vehicles = []
        self.rear_vehicles = []
        

class Vehicle():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = 0
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.yaw = 0

        self.lane_idx = 0
        self.dis_to_lane_tail = 0
        self.dis_to_lane_head = 0

class DynamicMap():
    def __init__(self, target_speed = 10):
        # events
        self.collision = False
        self.reached_goal = False
        self.lanes_updated = False

        # vehicle state
        self.ego_vehicle = Vehicle()

        # maps from opendrive
        #self.map = LocalMap()
        #self._map_trigger = threading.Event()

        #if self.map.setup_hdmap(file="", mtype='opendrive'): #MAP_UNKNOWN = 0 MAP_OPENDRIVE = 1 MAP_SUMO = 2
        #    self._map_trigger.set()
        
        self.lanes = []
        self.lanes_id = []
        
        self.first_route_update = False
        self.second_route_update = False

    def update_map(self, carla_world):
        self.init_dynamic_map()
        self.get_env_vehicle_information(carla_world)

        lane_nums = self.get_lane_information(carla_world)
        self.get_ego_vehicle_information(carla_world)

        print("[Dynamic_MAP] : ego pose",self.ego_vehicle.x, self.ego_vehicle.y)
        if lane_nums > 0:
            self.locate_ego_vehicle_in_lanes()
            self.locate_surrounding_objects_in_lanes(carla_world)

            
    def update_map_from_obs(self, obs, env):
        self.init_dynamic_map()

        # Ego information from obs
        self.ego_vehicle.x = obs[0]
        self.ego_vehicle.y = obs[1]
        self.ego_vehicle.vx = obs[2]
        self.ego_vehicle.vy = obs[3]
        self.ego_vehicle.lane_idx = 0

        self.ego_vehicle.v = math.sqrt(self.ego_vehicle.vx ** 2 + self.ego_vehicle.vy ** 2)
        # The control module used yaw and yawdt
        self.ego_vehicle.yaw = env.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        # self.ego_vehicle.yawdt = env.ego_vehicle.get_angular_velocity() 
        
        # Env information from obs
        for i in range(int(len(obs)/5-1)):
            vehicle = Vehicle()
            vehicle.x = obs[0+5+5*i]
            vehicle.y = obs[1+5+5*i]
            vehicle.vx = obs[2+5+5*i]
            vehicle.vy = obs[3+5+5*i]
            vehicle.lane_idx = 0
            vehicle.v = math.sqrt(vehicle.vx ** 2 + vehicle.vy ** 2)

            
            self.vehicles.append(vehicle)

        # print("[Dynamic Map]: Add env vehicles num", len(self.vehicles))
        if len(self.lanes) == 0:
            self.lanes.append(env.ref_path)
            self.lanes_id.append(0)
            self.lanes_updated = True
            env.real_time_ref_path_array = env.ref_path_array

        # For round driving
        # ego_to_start = math.sqrt((self.ego_vehicle.x - env.start_point.location.x) **2 +(self.ego_vehicle.y - env.start_point.location.y) **2)
        # ego_to_end = math.sqrt((self.ego_vehicle.x - env.goal_point.location.x) **2 +(self.ego_vehicle.y - env.goal_point.location.y) **2)
        # if ego_to_start < 10 and self.first_route_update == False:
        #     self.lanes = []
        #     self.lanes_id = []
        #     self.lanes.append(env.ref_path)
        #     self.lanes_id.append(0)
        #     self.lanes_updated = True
        #     self.first_route_update = True
        #     self.second_route_update = False
        #     env.real_time_ref_path_array = env.ref_path_array
            
        # if ego_to_end < 10 and self.second_route_update == False:
        #     self.lanes = []
        #     self.lanes_id = []
        #     self.lanes.append(env.ref_path2)
        #     self.lanes_id.append(0)
        #     self.lanes_updated = True
        #     self.second_route_update = True
        #     self.first_route_update = False
        #     env.real_time_ref_path_array = env.ref_path_array2

    def init_dynamic_map(self):
        self.vehicles = []
        self.lanes_updated = False

        for lane_id in range(len(self.lanes)):
            self.lanes[lane_id].front_vehicles.clear()


    def get_ego_vehicle_information(self, carla_world):
        actor_list = carla_world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] == "ego_vehicle":
                ego_vehicle = vehicle
                # ego vehicle
                self.ego_vehicle.x = ego_vehicle.get_location().x
                self.ego_vehicle.y = ego_vehicle.get_location().y
                self.ego_vehicle.v = math.sqrt(ego_vehicle.get_velocity().x ** 2 + ego_vehicle.get_velocity().y ** 2 + ego_vehicle.get_velocity().z ** 2)

                print("[TEST] : ego_vehicle speed:", self.ego_vehicle.v)

                self.ego_vehicle.yaw = ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
                self.ego_vehicle.yawdt = ego_vehicle.get_angular_velocity()

                self.ego_vehicle.vx = self.ego_vehicle.v * math.cos(self.ego_vehicle.yaw)
                self.ego_vehicle.vy = self.ego_vehicle.v * math.sin(self.ego_vehicle.yaw)
                if len(self.lanes) > 0:
                    dist_list = np.array([dist_from_point_to_polyline2d(
                                self.ego_vehicle.x,
                                self.ego_vehicle.y,
                                lane.central_path_array, return_end_distance=True)
                                for lane in self.lanes])

                    self.ego_vehicle.lane_idx = np.argmin(np.abs(dist_list[:, 0]))
                        
                self.dis_to_lane_tail = 0
                self.dis_to_lane_head = 0

    def get_env_vehicle_information(self, carla_world):
        self.vehicles = []
        # env vehicles
        actor_list = carla_world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for nv in vehicle_list:
            if nv.attributes['role_name'] != "ego_vehicle":
                vehicle = Vehicle()
                vehicle.x = nv.get_location().x
                vehicle.y = nv.get_location().y
                vehicle.v = math.sqrt(nv.get_velocity().x ** 2 + nv.get_velocity().y ** 2 + nv.get_velocity().z ** 2)

                vehicle.yaw = nv.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad

                vehicle.vx = vehicle.v * math.cos(vehicle.yaw)
                vehicle.vy = vehicle.v * math.sin(vehicle.yaw)
                self.vehicles.append(vehicle)

    def get_lane_information(self, carla_world, short_lane_thres=5):

        count_id = 0
        # Update new pose to map to read lanes
        self.map.receive_new_pose(self.ego_vehicle.x, self.ego_vehicle.y)

        # new_static_map = self.map.update()
        new_static_map = None # Used only reference path

        if new_static_map is not None:
            print("[Dynamic_MAP] : Update_lanes num = ",len(new_static_map.lanes))
            self.lanes = []
            self.lanes_id = []
            self.lanes_updated = True
            # The lanes will be updated if there are new lanes in static map
            if len(new_static_map.lanes) > 0:
                for path in new_static_map.lanes:
                    one_lane = Lane()
                    t_array = []
                    
                    # Extend the lane to avoid ego vehicle outside the start of the lane
                    dx = path.central_path_points[1].position.x - path.central_path_points[0].position.x
                    dy = path.central_path_points[1].position.y - path.central_path_points[0].position.y
                    for i in range(1,20,1):
                        lanepoint = Lanepoint()
                        lanepoint.position.x = path.central_path_points[0].position.x - (10 - i) * dx * 10
                        lanepoint.position.y = path.central_path_points[0].position.y - (10 - i) * dy * 10

                        one_lane.central_path.append(lanepoint)
                        waypoint_array = [lanepoint.position.x, lanepoint.position.y]
                        t_array.append(waypoint_array)
                    
                    for waypoint in path.central_path_points:
                        lanepoint = Lanepoint()
                        lanepoint.position.x = waypoint.position.x
                        lanepoint.position.y = waypoint.position.y

                        one_lane.central_path.append(lanepoint)
                        waypoint_array = [waypoint.position.x, waypoint.position.y]
                        t_array.append(waypoint_array)
                    one_lane.central_path_array = np.array(t_array)
                    road_len = polyline_length(np.array(one_lane.central_path_array))
                    if road_len <= short_lane_thres:
                        self.lanes_id.append(-1)
                        continue

                    one_lane.speed_limit = 60/3.6

                    #FIXME: Calculate id like this might be wrong
                    one_lane.lane_index = count_id
                    self.lanes.append(one_lane)
                    self.lanes_id.append(count_id)
                    count_id = count_id + 1

        # If there is no lanes in dynamic map, use reference path as the only lane
        if len(self.lanes) == 0:
            one_lane = Lane()
            t_array = []
            for waypoint in self.map._reference_path:
                lanepoint = Lanepoint()
                lanepoint.position.x = waypoint[0]
                lanepoint.position.y = waypoint[1]

                one_lane.central_path.append(lanepoint)
                t_array.append(waypoint)
            one_lane.central_path_array = np.array(t_array)
            one_lane.speed_limit = 30/3.6

            one_lane.lane_index = 0
            self.lanes.append(one_lane)
            self.lanes_id.append(0)
            self.lanes_updated = True


        return len(self.lanes)

    def locate_ego_vehicle_in_lanes(self, lane_end_dist_thres=5, lane_dist_thres=5):
        dist_list = np.array([dist_from_point_to_polyline2d(
            self.ego_vehicle.x, self.ego_vehicle.y,
            lane.central_path_array, return_end_distance=True)
            for lane in self.lanes])

        ego_lane_index = self.locate_object_in_lane()
        ego_lane_index_rounded = int(round(ego_lane_index))

        # print("ego_lane_index=",ego_lane_index)

        self.ego_vehicle.dis_to_lane_head = dist_list[:, 3]
        self.ego_vehicle.dis_to_lane_tail = dist_list[:, 4]


        if ego_lane_index_rounded < 0 or ego_lane_index_rounded > len(self.lanes) - 1:
            print("[Dynamic_map]: Ego_lane_index_error")
            return

        print("[Dynamic_map]: Distance to end", ego_lane_index, self.ego_vehicle.dis_to_lane_tail[ego_lane_index_rounded])


    def locate_surrounding_objects_in_lanes(self, carla_world, lane_dist_thres=1):
        
        lane_front_vehicle_list = [[] for _ in self.lanes]
        lane_rear_vehicle_list = [[] for _ in self.lanes]

        # TODO: separate vehicle and other objects?
        if self.vehicles is not None:
            for vehicle_idx, vehicle in enumerate(self.vehicles):
                dist_list = np.array([dist_from_point_to_polyline2d(
                    vehicle.x,
                    vehicle.y,
                    lane.central_path_array, return_end_distance=True)
                    for lane in self.lanes])
                closest_lane = np.argmin(np.abs(dist_list[:, 0]))

                # Determine if the vehicle is close to lane enough
                if abs(dist_list[closest_lane, 0]) > lane_dist_thres:
                    continue 
                if dist_list[closest_lane, 3] < self.ego_vehicle.dis_to_lane_head[closest_lane]:
                    # The vehicle is behind if its distance to lane start is smaller
                    lane_rear_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 3]))
                if dist_list[closest_lane, 4] < self.ego_vehicle.dis_to_lane_tail[closest_lane]:
                    # The vehicle is ahead if its distance to lane end is smaller
                    lane_front_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 4]))
                vehicle.lane_idx = closest_lane
        
        # Put the vehicles onto lanes
        for lane_id in range(len(self.lanes)):
            front_vehicles = np.array(lane_front_vehicle_list[lane_id])
            rear_vehicles = np.array(lane_rear_vehicle_list[lane_id])

            if len(front_vehicles) > 0:
                # Descending sort front objects by distance to lane end
                for vehicle_row in reversed(front_vehicles[:,1].argsort()):

                    front_vehicle_idx = int(front_vehicles[vehicle_row, 0])
                    front_vehicle = self.vehicles[front_vehicle_idx]
                    self.lanes[lane_id].front_vehicles.append(front_vehicle)

            if len(rear_vehicles) > 0:
                # Descending sort rear objects by distance to lane end
                for vehicle_row in reversed(rear_vehicles[:,1].argsort()):
                    rear_vehicle_idx = int(rear_vehicles[vehicle_row, 0])
                    rear_vehicle = self.vehicles[rear_vehicle_idx]
                    self.lanes[lane_id].rear_vehicles.append(rear_vehicle)
                
    def locate_object_in_lane(self, dist_list=None, lane_dist_thres=5):
        '''
        Calculate (continuous) lane index for a object.
        Parameters: dist_list is the distance buffer. If not provided, it will be calculated
        '''

        if not dist_list:
            dist_list = np.array([dist_from_point_to_polyline2d(
                self.ego_vehicle.x,
                self.ego_vehicle.y,
                lane.central_path_array) for lane in self.lanes])
        
        # Check if there's only two lanes
        if len(self.lanes) < 2:
            closest_lane = second_closest_lane = 0
        else:
            closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        # Signed distance from target to two closest lane
        closest_lane_dist, second_closest_lane_dist = dist_list[closest_lane, 0], dist_list[second_closest_lane, 0]

        if abs(closest_lane_dist) > lane_dist_thres:
            return -1 # TODO: return reasonable value

        # Judge whether the point is outside of lanes
        if closest_lane == second_closest_lane or closest_lane_dist * second_closest_lane_dist > 0:
            # The object is at left or right most
            return closest_lane
        else:
            # The object is between center line of lanes
            a, b = closest_lane, second_closest_lane
            la, lb = abs(closest_lane_dist), abs(second_closest_lane_dist)
            return (b*la + a*lb)/(lb + la)
            
            

        

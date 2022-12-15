import glob
import os
import sys
try:    	
    sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla')[0])
    
except IndexError:
	pass


import carla
import time
import numpy as np
import math
import random
import gym
import threading
from random import randint
from carla import Location, Rotation, Transform, Vector3D, VehicleControl
from collections import deque
from tqdm import tqdm
from gym import core, error, spaces, utils
from gym.utils import seeding
from agents.navigation.global_route_planner import GlobalRoutePlanner

from Agent.zzz.dynamic_map import Lanepoint, Lane, Vehicle
from Agent.zzz.tools import *
from Planning_library.coordinates import Coordinates

OBSTACLES_CONSIDERED = 3 # For t-intersection in Town02


global start_point
start_point = Transform()
start_point.location.x = 242
start_point.location.y = 120
start_point.location.z = 2
start_point.rotation.pitch = 0
start_point.rotation.yaw = -90
start_point.rotation.roll = 0

global ego_spawn_point
ego_spawn_point = Transform()
ego_spawn_point.location.x = 242
ego_spawn_point.location.y = 110
ego_spawn_point.location.z = 2
ego_spawn_point.rotation.pitch = 0
ego_spawn_point.rotation.yaw = -90
ego_spawn_point.rotation.roll = 0

global goal_point
goal_point = Transform()
goal_point.location.x = 245
goal_point.location.y = 29
goal_point.location.z = 0
goal_point.rotation.pitch = 0
goal_point.rotation.yaw = 0 
goal_point.rotation.roll = 0

global pedestrian_point
pedestrian_point = Transform()
pedestrian_point.location.x = 248
pedestrian_point.location.y = 80
pedestrian_point.location.z = 1
pedestrian_point.rotation.pitch = 0
pedestrian_point.rotation.yaw = 180 
pedestrian_point.rotation.roll = 0

class CarEnv_Town03_Complex:

    def __init__(self):
        
        # CARLA settings
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name != 'Carla/Maps/Town03':
            self.world = self.client.load_world('Town03')
        self.world.set_weather(carla.WeatherParameters(cloudiness=50, precipitation=30.0, sun_altitude_angle=60.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        self.dt = 0.05
        settings.fixed_delta_seconds = self.dt # Warning: When change simulator, the delta_t in controller should also be change.
        settings.substepping = True
        settings.max_substep_delta_time = 0.02  # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        settings.max_substeps = 10
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.free_traffic_lights(self.world)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_random_device_seed(0)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()
            
        # Debug setting
        self.debug = self.world.debug
        self.should_debug = True

        # Generate Reference Path
        self.global_routing()

        # RL settingss
        self.action_space = spaces.Discrete(11)
        self.low  = np.array([0]*20, dtype=np.float64)
        self.high = np.array([1]*20, dtype=np.float64)    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.action_dimension = 11
        self.state_dimension = 20

        # Ego Vehicle Setting
        global start_point
        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz_2020'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,0,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "ego_vehicle")
        self.ego_collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_vehicle = None
        self.stuck_time = None
        
        np.set_printoptions(suppress=True)

        # Walker
        self.set_fixed_vehicle_points()

        # Control Env Vehicle
        self.has_set = np.zeros(1000000)
        self.stopped_time = np.zeros(1000000)   

        # Record
        self.log_dir = "record.txt"
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0
        self.case_id = 0

       
    def free_traffic_lights(self, carla_world):
        traffic_lights = carla_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(5)
            tl.set_red_time(5)

    def global_routing(self):
        global goal_point
        global start_point

        start = start_point
        goal = goal_point
        print("Calculating route to x={}, y={}, z={}".format(
                goal.location.x,
                goal.location.y,
                goal.location.z))

        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1) # Carla 0913
        current_route = grp.trace_route(carla.Location(start.location.x,
                                                start.location.y,
                                                start.location.z),
                                carla.Location(goal.location.x,
                                                goal.location.y,
                                                goal.location.z))
        t_array = []
        self.ref_path = Lane()
        for wp in current_route:
            lanepoint = Lanepoint()
            lanepoint.position.x = wp[0].transform.location.x 
            lanepoint.position.y = wp[0].transform.location.y
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s

        ref_path_ori = convert_path_to_ndarray(self.ref_path.central_path)
        self.ref_path_array = dense_polyline2d(ref_path_ori, 2)
        self.ref_path_tangets = np.zeros(len(self.ref_path_array))
        
    def ego_vehicle_stuck(self, stay_thres = 2):           
        if self.stuck_time is None:
            self.stuck_time = time.time()
            return False
        
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity < 0.1:
            if time.time() - self.stuck_time > stay_thres:
                return True
        else:
            self.stuck_time = time.time()

        return False

    def ego_vehicle_pass(self):
        
        if self.ego_vehicle.get_location().y < 73.7:
            return True 
        
        return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self, use_ego_coordinate = True):

        state  = [] # np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
        state_ori  = [] #np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)

        # ego_vehicle_state = Vehicle()
        # ego_vehicle_state.x = self.ego_vehicle.get_location().x
        # ego_vehicle_state.y = self.ego_vehicle.get_location().y
        # ego_vehicle_state.v = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        # snapshot = self.world.get_snapshot()
        
        # accx = self.ego_vehicle.get_acceleration().x
        # accy = self.ego_vehicle.get_acceleration().y
        
        # # print("speed:", ego_vehicle_state.v, "acc:", accx, accy, snapshot.timestamp.elapsed_seconds)

        # ego_vehicle_state.yaw = self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        # ego_vehicle_state.yawdt = self.ego_vehicle.get_angular_velocity()

        # ego_vehicle_state.vx = ego_vehicle_state.v * math.cos(ego_vehicle_state.yaw)

        # ego_vehicle_state.vy = ego_vehicle_state.v * math.sin(ego_vehicle_state.yaw)

        # Ego state
        ego_state = [self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y,
                     self.ego_vehicle.get_velocity().x, self.ego_vehicle.get_velocity().y,
                     self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi]
        
        state_ori = state_ori + ego_state
        
        if use_ego_coordinate:
            ego_vehicle_coordiate = Coordinates(state_ori[0],state_ori[1],state_ori[4])
            state = state + list(ego_vehicle_coordiate.transfer_coordinate(state_ori[0],state_ori[1],
                                                                         state_ori[2],state_ori[3],
                                                                         state_ori[4]))
        
        for obj in self.state_obj:
            obj_state = [obj.get_location().x, obj.get_location().y,
                         obj.get_velocity().x, obj.get_velocity().y,
                         obj.get_transform().rotation.yaw / 180.0 * math.pi]
            
            state_ori = state_ori + obj_state
            
            if use_ego_coordinate:
                state = state + list(ego_vehicle_coordiate.transfer_coordinate(obj_state[0],obj_state[1],
                                                                         obj_state[2],obj_state[3],
                                                                         obj_state[4]))
            
        
        # state_ori[0] = ego_vehicle_state.x  
        # state_ori[1] = ego_vehicle_state.y 
        # state_ori[2] = ego_vehicle_state.vx 
        # state_ori[3] = ego_vehicle_state.vy 
        # state_ori[4] = ego_vehicle_state.yaw 

        # Obs state
        # actor_list = self.world.get_actors()
        # pedestrian_list = actor_list.filter("*walker.pedestrian*")
        
        # obs = pedestrian_list[0] 
        # state_ori[5] = obs.get_location().x
        # state_ori[6] = obs.get_location().y
        # state_ori[7] = obs.get_velocity().x
        # state_ori[8] = obs.get_velocity().y
        # state_ori[9] = obs.get_transform().rotation.yaw / 180.0 * math.pi
        
        # if use_ego_coordinate:
            
        #     ego_vehicle_coordiate = Coordinates(state_ori[0],state_ori[1],state_ori[4])                
        #     ego_state = ego_vehicle_coordiate.transfer_coordinate(state_ori[0],state_ori[1],
        #                                                                  state_ori[2],state_ori[3],
        #                                                                  state_ori[4])
            
        #     state = state + ego_state
            
        #     walker_state = ego_vehicle_coordiate.transfer_coordinate(state_ori[5],state_ori[6],
        #                                                                  state_ori[7],state_ori[8],
        #                                                                  state_ori[9])
        #     state[5] = walker_state[0]
        #     state[6] = walker_state[1]
        #     state[7] = walker_state[2]
        #     state[8] = walker_state[3]
        #     state[9] = walker_state[4]
        # else:
        #     state = state_ori

        return np.array(state), np.array(state_ori)

    def found_closest_obstacles_t_intersection(self, ego_ffstate):
        obs_tuples = []
        for obs in self.world.get_actors().filter('vehicle*'):
            # Calculate distance
            p1 = np.array([self.ego_vehicle.get_location().x ,  self.ego_vehicle.get_location().y])
            p2 = np.array([obs.get_location().x , obs.get_location().y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])
            
            # Obstacles too far
            one_obs = (obs.get_location().x, obs.get_location().y, obs.get_velocity().x, obs.get_velocity().y, obs.get_transform().rotation.yaw/ 180.0 * math.pi, p4)
            if 0 < p4 < 50:
                obs_tuples.append(one_obs)
        
        closest_obs = []
        fake_obs = [0 for i in range(11)]  #len(one_obs)
        for i in range(0, OBSTACLES_CONSIDERED ,1): # 3 obs
            closest_obs.append(fake_obs)
        
        # Sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[5])   
        for obs in sorted_obs:
            closest_obs[0] = obs 

        return closest_obs
                                            
    def record_information_txt(self):
        if self.task_num > 0:
            stuck_rate = float(self.stuck_num) / float(self.task_num)
            collision_rate = float(self.collision_num) / float(self.task_num)
            pass_rate = 1 - ((float(self.collision_num) + float(self.stuck_num)) / float(self.task_num))
            fw = open(self.log_dir, 'a')   
            # Write num
            fw.write(str(self.task_num)) 
            fw.write(", ")
            fw.write(str(self.case_id)) 
            fw.write(", ")
            fw.write(str(self.stuck_num)) 
            fw.write(", ")
            fw.write(str(self.collision_num)) 
            fw.write(", ")
            fw.write(str(stuck_rate)) 
            fw.write(", ")
            fw.write(str(collision_rate)) 
            fw.write(", ")
            fw.write(str(pass_rate)) 
            fw.write("\n")
            fw.close()               
            print("[CARLA]: Record To Txt: All", self.task_num, self.stuck_num, self.collision_num, self.case_id )

    def clean_task_nums(self):
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

    def reset(self):    

        # Ego vehicle
        self.state_obj = []
        self.spawn_ego_veh()
        self.spawn_human()
        self.spawn_fixed_veh()
        self.world.tick()
        
        # Stuck time
        self.stuck_time = None

        # State
        state, state_ori = self.wrap_state()
        self.last_state = state_ori

        # Record
        self.record_information_txt()
        self.task_num += 1
        self.case_id += 1
        
        self.driving_speed = []

        return state, state_ori

    def step(self, action, trust_RL = True):
        
        # Control ego vehicle
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.world.tick()

        # State
        state, state_ori = self.wrap_state()
        ego_speed = math.sqrt(state_ori[2]**2 + state_ori[3]**2)
        self.driving_speed.append(ego_speed)
        
        # Debug
        # if self.should_debug:
        #     ego_z = self.ego_vehicle.get_location().z
        #     start_loc = carla.Location(x = self.last_state[0],y = self.last_state[1],z = ego_z+0.1)
        #     end_loc = carla.Location(x = state_ori[0],y = state_ori[1],z = ego_z+0.1)
        #     if trust_RL:
        #         color = carla.Color(r=13, g=13, b=13, a=255)
        #         self.debug.draw_line(start_loc, end_loc, color=color, life_time = 10)
        #     else:
        #         self.debug.draw_line(start_loc, end_loc, life_time = 10)

        self.last_state = state_ori
        
        # Step reward
        v = math.sqrt(state_ori[2]**2 + state_ori[3]**2)
        reward = math.sqrt(v)*0.1
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            reward = -100
            AveSpeed = sum(self.driving_speed)/len(self.driving_speed)
            print("[CARLA]: Collision!, AveSpeed=", AveSpeed)
        
        if self.ego_vehicle_pass():
            done = True
            AveSpeed = sum(self.driving_speed)/len(self.driving_speed)
            print("[CARLA]: Successful!, AveSpeed=", AveSpeed)

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            reward = 0.0
            done = True
            AveSpeed = sum(self.driving_speed)/len(self.driving_speed)
            print("[CARLA]: Stuck!, AveSpeed=", AveSpeed)

        return state, reward, done, state_ori
    
    def draw_traj_HRL(self, trajectories, state_value, HRL_state_value, rule_act_idx):
        
        if self.should_debug:
            
            max_value = max(HRL_state_value)
            min_value = min(HRL_state_value)

            for j, trajectory in enumerate(trajectories):
                
                traj = trajectory[0]
                traj_value = (HRL_state_value[j+1] - min_value) / (max_value - min_value)
                
                min_thickness = 0.01
                max_thickness = 0.1
                value_thickness = float(min_thickness + (max_thickness - min_thickness)*traj_value)
                
                if j == rule_act_idx - 1:
                    color = carla.Color(r=255, g=0, b=0, a=255)
                else:
                    color = carla.Color(r=13, g=13, b=13, a=255)

                for i in range(len(traj.x)-1):
                    start_loc = carla.Location(x = traj.x[i],y = traj.y[i],z = 0.5)
                    end_loc = carla.Location(x = traj.x[i+1],y = traj.y[i+1],z = 0.5)
                    # self.debug.draw_arrow(start_loc, end_loc, thickness = value_thickness, arrow_size=0.01, color=color, life_time = 2)
        
            self.world.tick()
    
    def draw_traj(self, trajectories):
        
        if self.should_debug:
            for j, trajectory in enumerate(trajectories):
                traj = trajectory[0]
                thickness = 0.05
                for i in range(len(traj.x)-1):
                    start_loc = carla.Location(x = traj.x[i],y = traj.y[i],z = 0.5)
                    end_loc = carla.Location(x = traj.x[i+1],y = traj.y[i+1],z = 0.5)
                    self.debug.draw_arrow(start_loc, end_loc, thickness = thickness, arrow_size=0.01, life_time = 0.5)
            self.world.tick()

    def set_fixed_vehicle_points(self):
        
        self.spawn_env_vehicle_points = []

        # Vehicle 1
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        point = Transform()
        point.location.x = 246
        point.location.y = 110
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = -90
        point.rotation.roll = 0
        
        self.spawn_env_vehicle_points.append((point, vehicle_bp, False, False))
        
        # Vehicle 2
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        
        point = Transform()
        point.location.x = 246
        point.location.y = 100
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = -90
        point.rotation.roll = 0
        self.spawn_env_vehicle_points.append((point, vehicle_bp, False, False))
        
        # Vehicle 3
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.carlamotors.firetruck')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        
        point = Transform()
        point.location.x = 240
        point.location.y = 80
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = -90
        point.rotation.roll = 0
        self.spawn_env_vehicle_points.append((point, vehicle_bp, True, True))
        
        # Vehicle 4
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.mini.cooper_s')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        
        point = Transform()
        point.location.x = 240
        point.location.y = 110
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = -90
        point.rotation.roll = 0
        self.spawn_env_vehicle_points.append((point, vehicle_bp, True, True))
        
        # Vehicle 5
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        
        point = Transform()
        point.location.x = 233
        point.location.y = 90
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = 90
        point.rotation.roll = 0
        self.spawn_env_vehicle_points.append((point, vehicle_bp, True, False))
        
        # Vehicle 6
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.audi.tt')[0]
        if vehicle_bp.has_attribute('color'):
            color = '255,255,255'
            vehicle_bp.set_attribute('color', color)
        
        point = Transform()
        point.location.x = 230
        point.location.y = 110
        point.location.z = 2
        point.rotation.pitch = 0
        point.rotation.yaw = 90
        point.rotation.roll = 0
        self.spawn_env_vehicle_points.append((point, vehicle_bp, True, False))
        
    def spawn_fixed_veh(self):
        
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        
        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] != "ego_vehicle" :
                vehicle.destroy()
                
        for transform, vehicle_bp, autopilot, in_state in self.spawn_env_vehicle_points:
            vehicle = self.world.spawn_actor(vehicle_bp, transform)
            vehicle.set_autopilot(enabled = autopilot)
            if in_state:
                self.state_obj.append(vehicle)
            
    def spawn_human(self):
        
        actor_list = self.world.get_actors()
        walker_list = actor_list.filter("*walker*")
        
        for walker in walker_list:
            walker.destroy()
        
        blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_bp = blueprintsWalkers[7]#random.choice(blueprintsWalkers)
        
        global pedestrian_point
        walker = self.world.spawn_actor(walker_bp, pedestrian_point)
        
        walkercontrol = carla.WalkerControl()
        walkercontrol.speed = 0.9
        pedestrain_heading = 180
        walkercontrol.direction = carla.Rotation(0,pedestrain_heading,0).get_forward_vector()
        
        walker.apply_control(walkercontrol)
        
        self.state_obj.append(walker)

    def spawn_ego_veh(self):
        global ego_spawn_point
        if self.ego_vehicle is not None:
            self.ego_collision_sensor.destroy()
            self.ego_vehicle.destroy()

        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, ego_spawn_point)
        self.ego_collision_sensor = self.world.spawn_actor(self.ego_collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False
        
        for i in range(20):
            self.world.tick()
    
    
    # vehicle.audi.a2
    # vehicle.audi.tt
    # vehicle.micro.microlino
    # vehicle.dodge
    # vehicle.jeep.wrangler_rubicon
    # vehicle.mini.cooper_s
    # vehicle.nissan.micra
    # vehicle.mercedes.coupe
    # vehicle.seat.leon
    # vehicle.toyota.prius
    # vehicle.harley-davidson.low_rider
    # vehicle.carlamotors.carlacola
    # vehicle.mercedes.coupe_2020
    # vehicle.volkswagen.t2_2021
    # vehicle.bh.crossbike
    # vehicle.mini.cooper_s_2021
    # vehicle.dodge.charger_police_2020
    # vehicle.bmw.grandtourer
    # vehicle.mercedes.sprinter
    # vehicle.vespa.zx125
    # vehicle.yamaha.yzf
    # vehicle.ford.crown
    # vehicle.ford.ambulance
    # vehicle.nissan.patrol_2021
    # vehicle.nissan.patrol
    # vehicle.kawasaki.ninja
    # vehicle.lincoln.mkz_2020
    # vehicle.ford.mustang
    # vehicle.lincoln.mkz_2017
    # vehicle.tesla.cybertruck
    # vehicle.chevrolet.impala
    # vehicle.volkswagen.t2
    # vehicle.citroen.c3
    # vehicle.diamondback.century
    # vehicle.tesla.model3
    # vehicle.carlamotors.firetruck
    # vehicle.audi.etron
    # vehicle.dodge.charger_2020
    # vehicle.gazelle.omafiets

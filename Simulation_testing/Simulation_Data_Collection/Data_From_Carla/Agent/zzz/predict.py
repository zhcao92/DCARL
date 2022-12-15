import numpy as np
import copy
import math
from Agent.zzz.frenet import Frenet_path

class predict():
    def __init__(self, dynamic_map, considered_obs_num, maxt, dt, robot_radius, radius_speed_ratio, move_gap, ego_speed):

        self.maxt = maxt
        self.dt = dt
        self.check_radius = robot_radius #  + radius_speed_ratio * ego_speed
        self.move_gap = move_gap
        self.dynamic_map = dynamic_map

        try:
            interested_vehicles = self.found_interested_vehicles(considered_obs_num)
            self.predict_paths = self.prediction_obstacle(interested_vehicles, self.maxt, self.dt)
        except:
            self.predict_paths = []

    def check_collision(self, fp):
        
        if len(self.predict_paths) == 0 or len(fp.t) < 2 :
            return True
            
        # two circles for a vehicle
        # fp_front = copy.deepcopy(fp)
        # fp_back = copy.deepcopy(fp)
        
        # fp_front.x = (np.array(fp.x)+np.cos(np.array(fp.yaw))*self.move_gap).tolist()
        # fp_front.y = (np.array(fp.y)+np.sin(np.array(fp.yaw))*self.move_gap).tolist()
        # print("fp_front.x",fp_front.x)
        # print("fp_front.y",fp_front.y)


        # fp_back.x = (np.array(fp.x)-np.cos(np.array(fp.yaw))*self.move_gap).tolist()
        # fp_back.y = (np.array(fp.y)-np.sin(np.array(fp.yaw))*self.move_gap).tolist()

        # for path in self.predict_paths:
        #     len_predict_t = min(len(fp.x)-1, len(path.t)-1)
        #     predict_step = 2
        #     start_predict = 2
        #     for t in range(start_predict, len_predict_t, predict_step):
        #         d = (path.x[t] - fp_front.x[t])**2 + (path.y[t] - fp_front.y[t])**2
        #         if d <= self.check_radius**2: 
        #             return False
        #         d = (path.x[t] - fp_back.x[t])**2 + (path.y[t] - fp_back.y[t])**2
        #         if d <= self.check_radius**2: 
        #             return False
                
        # For Pedestrain
        for path in self.predict_paths:
            len_predict_t = min(len(fp.x)-1, len(path.t)-1)
            predict_step = 2
            start_predict = 2
            for t in range(start_predict, len_predict_t, predict_step):
                d = (path.x[t] - fp.x[t])**2 + (path.y[t] - fp.y[t])**2
                if d <= self.check_radius**2: 
                    return False
        return True

    def found_interested_vehicles(self, interested_vehicles_num=3):

        interested_vehicles = []

        # Get interested vehicles by distance
        distance_tuples = []
        ego_loc = np.array([self.dynamic_map.ego_vehicle.x,self.dynamic_map.ego_vehicle.y])

        for vehicle_idx, vehicle in enumerate(self.dynamic_map.vehicles): 
            vehicle_loc = np.array([vehicle.x, vehicle.y])
            d = np.linalg.norm(vehicle_loc - ego_loc)

            distance_tuples.append((d, vehicle_idx))
            
        sorted_vehicle = sorted(distance_tuples, key=lambda vehicle_dis: vehicle_dis[0])

        for _, vehicle_idx in sorted_vehicle:
            interested_vehicles.append(self.dynamic_map.vehicles[vehicle_idx])
            if len(interested_vehicles) >= interested_vehicles_num:
                break
        return interested_vehicles

    def prediction_obstacle(self, vehicles, max_prediction_time, delta_t): 
        predict_paths = []
        for vehicle in vehicles:

            predict_path_front = Frenet_path()
            predict_path_back = Frenet_path()
            predict_path_front.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            predict_path_back.t = [t for t in np.arange(0.0, max_prediction_time, delta_t)]
            ax = 0 #one_ob[9]
            ay = 0 #one_ob[10]
            # print("vehicle information",vehicle.x, vehicle.y, vehicle.vx, vehicle.vy, vehicle.yaw)

            vx_predict = vehicle.vx*np.ones(len(predict_path_front.t))
            vy_predict = vehicle.vy*np.ones(len(predict_path_front.t))

            x_predict = vehicle.x + np.arange(len(predict_path_front.t))*delta_t*vx_predict
            y_predict = vehicle.y + np.arange(len(predict_path_front.t))*delta_t*vy_predict
            
            predict_path_front.x = (x_predict + math.cos(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_front.y = (y_predict + math.sin(vehicle.yaw)*np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_back.x = (x_predict - math.cos(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
            predict_path_back.y = (y_predict - math.sin(vehicle.yaw)*np.ones(len(predict_path_back.t))*self.move_gap).tolist()
        
            predict_paths.append(predict_path_front)
            predict_paths.append(predict_path_back)

        return predict_paths

    


import numpy as np
import math
import time

from collections import deque
from Agent.zzz.actions import ControlAction
from Agent.zzz.tools import *



class Controller(object):

    def __init__(self):
        self._lon_controller = LonController()
        self._lat_controller = PurePuesuitController()

    def get_control(self, dynamic_map, trajectory, target_speed):

        ego_vehicle = dynamic_map.ego_vehicle

        acc = self._lon_controller.run_step(target_speed[-1], dynamic_map.ego_vehicle.v)
        steering = self._lat_controller.run_step(ego_vehicle, trajectory)

        return ControlAction(acc,steering)

class LonController(object):

    def __init__(self):
        """
        vehicle: actor to apply to local planner logic onto
        K_P: Proportional term
        K_D: Differential term
        K_I: Integral term
        dt: time differential in seconds
        """
        self._K_P = 0.25/3.6
        self._K_D = 0#.01
        self._K_I = 0#0.012 #FIXME: To stop accmulate error when repectly require control signal in the same state.
        self._dt = 0.1 # TODO: timestep
        self._integ = 0.0
        self._e_buffer = deque(maxlen=30)

    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

        target_speed: target speed in Km/h
        return: throttle control in the range [0, 1]
        """

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_speed:  target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [-1, 1]
        """
        if target_speed == 0:
            return -1

        target_speed = target_speed*3.6
        current_speed = current_speed*3.6

        _e = (target_speed - current_speed)
        self._integ += _e * self._dt
        self._e_buffer.append(_e)

        if current_speed < 2:
            self._integ = 0

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = self._integ

        else:
            _de = 0.0
            _ie = 0.0
        kp = self._K_P
        ki = self._K_I
        kd = self._K_D

        if target_speed < 5:
            ki = 0
            kd = 0
        
        calculate_value = np.clip((kp * _e) + (kd * _de) + (ki * _ie), -1.0, 1.0)
        return calculate_value

class PurePuesuitController(object):

    def __init__(self):
        pass

    def run_step(self, ego_vehicle, trajectory):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.
        """

        current_speed = ego_vehicle.v
        control_point = self._control_point(ego_vehicle, trajectory, current_speed)
        if len(control_point) < 2:
            return 0.0
        return self._purepersuit_control(control_point, ego_vehicle)

    def _control_point(self, ego_vehicle, trajectory, current_speed, resolution = 0.1):

        if current_speed > 10:        
            control_target_dt = 0.5 - (current_speed - 10)*0.01
        else:
            control_target_dt = 0.5
        # control_target_dt = 0.4
     
        control_target_distance = control_target_dt * current_speed  ## m
        if control_target_distance < 3:
            control_target_distance = 3


        ego_loc = np.array([ego_vehicle.x, ego_vehicle.y])
        
        # trajectory_dense = trajectory  #FIXME
        trajectory_dense = dense_polyline2d(trajectory, resolution)

        end_idx = self.get_next_idx(ego_loc, trajectory_dense, control_target_distance)
        wp_loc = trajectory_dense[end_idx]

        return wp_loc


    def _purepersuit_control(self, waypoint, ego_vehicle):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        
        ego_yaw = ego_vehicle.yaw 

        ego_x = ego_vehicle.x
        ego_y = ego_vehicle.y
        
        v_vec = np.array([math.cos(ego_yaw),
                          math.sin(ego_yaw),
                          0.0])

        target_x = waypoint[0]
        target_y = waypoint[1]

        w_vec = np.array([target_x-
                          ego_x, target_y -
                          ego_y, 0.0])


        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                         (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        lf = 1.2
        lr = 1.95
        lwb = lf + lr
        
        v_rear_x = ego_x - v_vec[0]*lr/np.linalg.norm(v_vec)
        v_rear_y = ego_y - v_vec[1]*lr/np.linalg.norm(v_vec)
        l = (target_x-v_rear_x)*(target_x-v_rear_x)+(target_y-v_rear_y)*(target_y-v_rear_y)
        l = math.sqrt(l)

        theta = np.arctan(2*np.sin(_dot)*lwb/l)
        
        k = 1.0 # XXX: np.pi/180*50
        theta = theta * k
        return theta


    # TODO: Move into library
    def convert_trajectory_to_ndarray(self, trajectory):
        trajectory_array = [(pose.pose.position.x, pose.pose.position.y) for pose in trajectory.poses]
        return np.array(trajectory_array)

    def get_idx(self, loc, trajectory):
        dist = np.linalg.norm(trajectory-loc,axis=1)
        idx = np.argmin(dist)
        return idx

    def get_next_idx(self, start_loc, trajectory, distance):

        start_idx = self.get_idx(start_loc,trajectory)
        dist_list = np.cumsum(np.linalg.norm(np.diff(trajectory,axis = 0),axis = 1))
        for end_idx in range(start_idx,len(trajectory)-1):
            if dist_list[end_idx] > dist_list[start_idx] + distance:
                return end_idx

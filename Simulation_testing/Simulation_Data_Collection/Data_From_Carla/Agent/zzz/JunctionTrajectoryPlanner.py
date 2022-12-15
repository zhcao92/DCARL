import numpy as np
import bisect
import copy
import time

from Agent.zzz.tools import *
from Agent.zzz.predict import predict
from Agent.zzz.frenet import *
from Agent.zzz.actions import TrajectoryAction
from Agent.zzz.cubic_spline_planner import Spline2D


# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 500.0  # maximum curvature [1/m]
MAX_LEFT_WIDTH = -4
MAX_RIGHT_WIDTH = 4
D_ROAD_W = 2  # road width sampling length [m]
DT = 0.3  # time tick [s]
MAXT = 4.2  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 15 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed

# Collision check
OBSTACLES_CONSIDERED = 4
ROBOT_RADIUS = 1  # robot radius [m]
RADIUS_SPEED_RATIO = 1 # higher speed, bigger circle
MOVE_GAP = 1
ONLY_SAMPLE_TO_LEFT = True

# Cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0
KRLS = 1.0


class JunctionTrajectoryPlanner(object):

    def __init__(self, target_line = 0):
        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()
        self.reference_path = None

        self._dynamic_map = None
        self.ref_path = None
        self.ref_path_tangets = None
        self.ob = None
        self.csp = None
        self.dist_to_end = [0,0,0,0,0]
        self.target_line = target_line

        self.radius = ROBOT_RADIUS
        self.move_gap = MOVE_GAP
        self.target_speed = TARGET_SPEED
        self.dts = D_T_S
    
    def clear_buff(self, clean_csp=True):

        if self.csp is None:
            return

        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()
        self.reference_path = None
        if clean_csp:
            self.csp = None
            
    def build_frenet_path(self, dynamic_map, clean_current_csp = False):

        if self.csp is None or clean_current_csp:
            self.reference_path = dynamic_map.lanes[0].central_path # When dynamic map has one ref path in lane
            ref_path_ori = convert_path_to_ndarray(self.reference_path)
            self.ref_path = dense_polyline2d(ref_path_ori, 2)
            self.ref_path_tangets = np.zeros(len(self.ref_path))

            Frenetrefx = self.ref_path[:,0]
            Frenetrefy = self.ref_path[:,1]
            tx, ty, tyaw, tc, self.csp = self.generate_target_course(Frenetrefx,Frenetrefy)
    
    def trajectory_update(self, dynamic_map):
        
        self.all_trajectory = self.get_all_candidate_trajectories(dynamic_map)
        
        if self.all_trajectory is None:
            return None
        
        index = self.get_optimal_trajectory(self.all_trajectory)
        final_trajectory = self.get_trajectory_by_index(index, self.all_trajectory)
        self.update_last_trajectory(final_trajectory)
        
        return self.convert_trajectory_to_TAction(final_trajectory), index
        
    def get_all_candidate_trajectories(self, dynamic_map):
        
        if not self.initialize(dynamic_map):
            return None
        
        start_state = self.calculate_start_state(dynamic_map)
        csp = self.csp
        c_speed = self.c_speed
        
        fplist = self.calc_frenet_paths(c_speed, start_state)
        fplist = self.calc_global_paths(fplist, csp)
        path_tuples = []
        i = 0
        for fp in fplist:
            one_path = [fp, fp.cf, i]
            i = i + 1
            path_tuples.append(one_path)
        
        return path_tuples
    
    def get_optimal_trajectory(self, path_tuples):
            
        sorted_fplist = sorted(path_tuples, key=lambda path_tuples: path_tuples[1])
        sorted_fplist = self.check_paths(sorted_fplist)
        for fp, score, index in sorted_fplist:
            if self.obs_prediction.check_collision(fp):
                return index + 1 # 0 for brake trajectory
        return 0
    
    def get_trajectory_by_index(self, act_idx, all_trajectory):
            
        if act_idx == 0:
            backup_trajectory, backup_index = self.get_backup_trajectory(all_trajectory)
            traj = backup_trajectory
        else:
            traj = all_trajectory[int(act_idx - 1)][0]
            
        return traj
        
    def get_backup_trajectory(self, all_trajectory):
        
        if len(self.last_trajectory_rule.s_d) > 5 and self.c_speed > 1:
            backup_trajectory = self.last_trajectory_rule
            backup_trajectory.s_d = [0] * len(self.last_trajectory_rule.s_d)
        else:
            sorted_fplist = sorted(all_trajectory, key=lambda all_trajectory: all_trajectory[1])
            backup_trajectory =  sorted_fplist[0][0]
            backup_trajectory.s_d = [0] * len(backup_trajectory.s_d)
 
        return backup_trajectory, 0
    
    def convert_trajectory_to_TAction(self, trajectory):
        
        return TrajectoryAction(np.c_[trajectory.x, trajectory.y], trajectory.s_d)
    
    def update_last_trajectory(self, trajectory):
        
        self.last_trajectory_array_rule = np.c_[trajectory.x, trajectory.y]
        self.last_trajectory_rule = trajectory 
        
    # def trajectory_update_CP(self, CP_action, rule_trajectory, update=True):
        
    #     if CP_action == 0:
    #         # print("[CP]:----> Brake") 
    #         generated_trajectory =  self.all_trajectory[0][0]
    #         trajectory_array = np.c_[generated_trajectory.x, generated_trajectory.y]
    #         trajectory_action = TrajectoryAction(trajectory_array, [0] * len(trajectory_array))
    #         return trajectory_action       
        
           
    #     fplist = self.all_trajectory
    #     bestpath = fplist[int(CP_action - 1)][0]
    #     bestpath.s_d
    #     trajectory_array = np.c_[bestpath.x, bestpath.y]
        
    #     # next time when you calculate start state
    #     if update==True:
    #         self.last_trajectory_array_rule = trajectory_array
    #         self.last_trajectory_rule = bestpath 

    #     trajectory_action = TrajectoryAction(trajectory_array, bestpath.s_d[:len(trajectory_array)])
    #     # print("[CP]: ------> CP Successful Planning")     
              
    #     return trajectory_action

    def initialize(self, dynamic_map):
        self._dynamic_map = dynamic_map

        try:
            # calculate dist to the end of ref path 
            if self.ref_path is not None:
                self.dist_to_end = dist_from_point_to_polyline2d(dynamic_map.ego_vehicle.x, dynamic_map.ego_vehicle.y, self.ref_path, return_end_distance=True)

            # estabilish frenet frame
            if self.csp is None or dynamic_map.lanes_updated == True: # or self.dist_to_end[4] < 10 or self.dist_to_end[0] > 20:
                self.last_trajectory_array_rule = np.c_[0, 0]
                self.build_frenet_path(dynamic_map, True)   
        
            # initialize prediction module
            self.obs_prediction = predict(dynamic_map, OBSTACLES_CONSIDERED, MAXT, DT, self.radius, RADIUS_SPEED_RATIO, self.move_gap,
                                        dynamic_map.ego_vehicle.v)

            return True

        except:
            # print("[UBP]: ------> Initialize fail ")
            return False

    def ref_tail_speed(self, dynamic_map, desired_speed):

        velocity = []
        for i in range(len(desired_speed)):

            if self.ref_path is None:
                velocity.append(0)
                velocity.append(0)
                velocity.append(0)
                velocity.append(0)
                velocity.append(0)
                continue
                
            # if self.dist_to_end[4] <= 5:
            #     velocity.append(0)
            #     continue

            dec = 0.1

            available_speed = math.sqrt(2*dec*self.dist_to_end[4]) # m/s
            ego_v = dynamic_map.ego_vehicle.v

            if available_speed > ego_v:
                velocity.append(desired_speed[i])
                continue

            dt = 0.2
            vehicle_dec = (ego_v - available_speed)*10
            tail_speed = ego_v - vehicle_dec*dt
            if desired_speed[i] > tail_speed:
                velocity.append(max(0, tail_speed))

        return velocity

    def calculate_start_state(self, dynamic_map):
        start_state = Frenet_state()
        # print("init",start_state.s0,start_state.c_d,self.c_speed)

        # if len(self.last_trajectory_array_rule) > 5:
        #     # find closest point on the last trajectory
        #     mindist = float("inf")
        #     bestpoint = 0
        #     for t in range(len(self.last_trajectory_rule.x)):
        #         pointdist = (self.last_trajectory_rule.x[t] - dynamic_map.ego_vehicle.x) ** 2 + (self.last_trajectory_rule.y[t] - dynamic_map.ego_vehicle.y) ** 2
        #         if mindist >= pointdist:
        #             mindist = pointdist
        #             bestpoint = t
        #     start_state.s0 = self.last_trajectory_rule.s[bestpoint]
        #     start_state.c_d = self.last_trajectory_rule.d[bestpoint]
        #     start_state.c_d_d = self.last_trajectory_rule.d_d[bestpoint]
        #     start_state.c_d_dd = self.last_trajectory_rule.d_dd[bestpoint]
        #     self.c_speed = dynamic_map.ego_vehicle.v

        # else:
        self.c_speed = dynamic_map.ego_vehicle.v

        ffstate = get_frenet_state(dynamic_map.ego_vehicle, self.ref_path, self.ref_path_tangets)

        start_state.s0 = ffstate.s 
        start_state.c_d = -ffstate.d # current lateral position [m]
        start_state.c_d_d = ffstate.vd # current lateral speed [m/s]
        start_state.c_d_dd = 0   # current latral acceleration [m/s]

            
        return start_state
    

    def generate_target_course(self, x, y):
        csp = Spline2D(x, y)
        s = np.arange(0, csp.s[-1], 0.1)

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = csp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(csp.calc_yaw(i_s))
            rk.append(csp.calc_curvature(i_s))

        return rx, ry, ryaw, rk, csp

    def calc_frenet_paths(self, c_speed, start_state): # input state

        frenet_paths = []

        s0 = start_state.s0
        c_d = start_state.c_d
        c_d_d = start_state.c_d_d
        c_d_dd = start_state.c_d_dd

        # generate path to each offset goal
        # left_sample_bound = 4 * MAX_ROAD_WIDTH 
        for di in np.arange(MAX_LEFT_WIDTH, MAX_RIGHT_WIDTH+1, D_ROAD_W):

            # Lateral motion planning
            for Ti in np.arange(MINT, MAXT, DT):
                fp = Frenet_path()

                lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti) 

                fp.t = np.arange(0.0, Ti, DT).tolist() # [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]                        
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Loongitudinal motion planning (Velocity keeping)
                for tv in np.arange(self.target_speed - self.dts * N_S_SAMPLE, self.target_speed + self.dts * N_S_SAMPLE, self.dts):
                    # print("werling_sample",tv, Ti, di)
                    tfp = copy.deepcopy(fp)
                    lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (self.target_speed - tfp.s_d[-1])**2

                    tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                    tfp.cv = KJ * Js + KT * Ti + KD * ds
                    tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def calc_global_paths(self, fplist, csp):

        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                iyaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(iyaw + math.pi / 2.0)
                fy = iy + di * math.sin(iyaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            dx = np.diff(np.array(fp.x))
            dy = np.diff(np.array(fp.y))
            fp.yaw = np.arctan2(dy,dx).tolist()
            fp.ds = np.sqrt(dx**2+dy**2).tolist()
            
            try:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            except:
                fp.yaw.append(0.1)
                fp.ds.append(0.1)

            # calc curvature
            # fp.c = (np.diff(fp.yaw) / np.array(fp.ds)).tolist()

            for i in range(len(fp.yaw) - 1):
                # carla simulation bug
                if fp.ds[i]<0.00001:
                    fp.ds[i] = 0.1
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_paths(self, fplist):
        okind = []
        for i, _ in enumerate(fplist):

            if any([v > MAX_SPEED for v in fplist[i][0].s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i][0].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in fplist[i][0].c]):  # Max curvature check
                continue

            okind.append(i)

        return [fplist[i] for i in okind]


class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # s=start  e=end

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:

    def __init__(self):
        self.t = [] # time
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

class Frenet_state:

    def __init__(self):
        self.t = 0.0
        self.d = 0.0
        self.d_d = 0.0
        self.d_dd = 0.0
        self.d_ddd = 0.0
        self.s = 0.0
        self.s_d = 0.0
        self.s_dd = 0.0
        self.s_ddd = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.ds = 0.0
        self.c = 0.0

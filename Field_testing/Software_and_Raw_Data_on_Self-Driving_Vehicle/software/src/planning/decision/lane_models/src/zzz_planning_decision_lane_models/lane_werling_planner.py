
import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import math

from cubic_spline_planner import Spline2D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from zzz_navigation_msgs.msg import Lane
from zzz_driver_msgs.utils import get_speed
from zzz_cognition_msgs.msg import RoadObstacle
from zzz_common.kinematics import get_frenet_state
from zzz_common.geometry import dense_polyline2d
from zzz_planning_msgs.msg import DecisionTrajectory

from zzz_planning_decision_continuous_models.common import rviz_display, convert_ndarray_to_pathmsg, convert_path_to_ndarray
from zzz_planning_decision_continuous_models.predict import predict

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 500.0  # maximum curvature [1/m]
# MAX_ROAD_WIDTH = 4.0   # maximum road width [m] # related to RL action space
# LEFT_SAMPLE_BOUND = 4.0
# RIGHT_SAMPLE_BOUND = 4.0
D_ROAD_W = 1  # road width sampling length [m]
DT = 0.3  # time tick [s]
MAXT = 8.6  # max prediction time [m]
MINT = 8.0  # min prediction time [m]
D_T_S = 2 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed

# collision check
OBSTACLES_CONSIDERED = 5
ROBOT_RADIUS = 2.4  # robot radius [m]
RADIUS_SPEED_RATIO = 0.0 # higher speed, bigger circle
MOVE_GAP = 1.0

# Cost weights
KJ = 0.1 * 2
KT = 0.1 * 0.5
KD = 1.0
KLAT = 1.0
KLON = 1.0


class Werling(object):

    def __init__(self, target_line, lane_idx, lane_num, lane_width = 3):

        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()
        # self.reference_path = None

        self.rviz_all_trajectory = None

        # self._dynamic_map = None
        self.ref_path = None
        self.ref_path_tangets = None
        self.ob = None
        self.csp = None
        self.target_speed = 0

        self._lane_num = lane_num
        self._lane_idx = lane_idx
        self._lane_width = lane_width

        self.rivz_element = rviz_display()

        self.build_frenet_path(target_line)
    
    def lane_change_clean_buff(self):
        
        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()

    def build_frenet_path(self, target_line):

        if self.csp is None:
            ref_path_ori = convert_path_to_ndarray(target_line)
            self.ref_path = dense_polyline2d(ref_path_ori, 2)
            self.ref_path_tangets = np.zeros(len(self.ref_path))

            Frenetrefx = self.ref_path[:,0]
            Frenetrefy = self.ref_path[:,1]
            tx, ty, tyaw, tc, self.csp = self.generate_target_course(Frenetrefx,Frenetrefy)
    
    def trajectory_update(self, dynamic_map, target_speed, ego_lane_index):
        
        if self.initialize_obj_predict(dynamic_map):
            self.all_trajectory = []
            self._ego_lane_index = ego_lane_index
            start_state = self.calculate_start_state(dynamic_map)
            generated_trajectory, high_resolution_speed = self.frenet_optimal_planning(self.csp, self.c_speed, start_state, target_speed)

            if generated_trajectory is not None:
                local_desired_speed = high_resolution_speed #generated_trajectory.s_d[-1]
                trajectory_array = np.c_[generated_trajectory.x, generated_trajectory.y]
                self.last_trajectory_array_rule = trajectory_array
                self.last_trajectory_rule = generated_trajectory
                rospy.logdebug("----> Lane_Werling: Successful Planning")
            else:
                # generated_trajectory = best_free_fp
                # local_desired_speed = 0
                # trajectory_array = np.c_[generated_trajectory.x, generated_trajectory.y]
                # self.last_trajectory_array_rule = trajectory_array
                # self.last_trajectory_rule = generated_trajectory
                rospy.logdebug("----> Lane_Werling: Fail to find a solution")
         
            self.rivz_element.candidates_trajectory = self.rivz_element.put_trajectory_into_marker(self.all_trajectory)
            self.rivz_element.prediciton_trajectory = self.rivz_element.put_trajectory_into_marker(self.obs_prediction.obs_paths)
            self.rivz_element.collision_circle = self.obs_prediction.rviz_collision_checking_circle
            return trajectory_array, local_desired_speed
        else:
            return None, None

    def initialize_obj_predict(self, dynamic_map):

        try:
            if self.csp is None:
                self.build_frenet_path()
            # initialize prediction module
            self.obs_prediction = predict(dynamic_map, OBSTACLES_CONSIDERED, MAXT, DT, ROBOT_RADIUS, RADIUS_SPEED_RATIO, MOVE_GAP,
                                        get_speed(dynamic_map.ego_state))
            return True
        except:
            return False

    def calculate_start_state(self, dynamic_map):
        start_state = Frenet_state()

        if len(self.last_trajectory_array_rule) > 5:
            # find closest point on the last trajectory
            mindist = float("inf")
            bestpoint = 0
            for t in range(len(self.last_trajectory_rule.x)):
                pointdist = (self.last_trajectory_rule.x[t] - dynamic_map.ego_state.pose.pose.position.x) ** 2 + (self.last_trajectory_rule.y[t] - dynamic_map.ego_state.pose.pose.position.y) ** 2
                if mindist >= pointdist:
                    mindist = pointdist
                    bestpoint = t
            start_state.s0 = self.last_trajectory_rule.s[bestpoint]
            start_state.c_d = self.last_trajectory_rule.d[bestpoint]
            start_state.c_d_d = self.last_trajectory_rule.d_d[bestpoint]
            start_state.c_d_dd = self.last_trajectory_rule.d_dd[bestpoint]
            # self.c_speed = self.last_trajectory_rule.s_d[bestpoint]
            ego_state = dynamic_map.ego_state
            self.c_speed = get_speed(ego_state) 
        else:
            ego_state = dynamic_map.ego_state
            self.c_speed = get_speed(ego_state)       # current speed [m/s]
            ffstate = get_frenet_state(dynamic_map.ego_state, self.ref_path, self.ref_path_tangets)

            start_state.s0 = ffstate.s 
            start_state.c_d = -ffstate.d # current lateral position [m]
            start_state.c_d_d = ffstate.vd # current lateral speed [m/s]
            start_state.c_d_dd = 0   # current latral acceleration [m/s]
            
        return start_state


    def frenet_optimal_planning(self, csp, c_speed, start_state, low_resolution_speed):

        # No adjustment
        di_range = np.array([0])
        Ti_range = np.arange(MINT, MAXT, DT)
        vi_range = np.array([max(10/3.6, low_resolution_speed)])
        best_free_fp, fp_available = self.calculate_path_in_given_range(csp, c_speed, start_state, 
                                                                            di_range, Ti_range, vi_range)
        
        if fp_available:
            rospy.logdebug("----> Lane_Werling: Successful Planning (No adjustment)")
            return best_free_fp, low_resolution_speed


        # Lateral slight adjustment
        right_bound = - (max((self._lane_idx - self._ego_lane_index),0)*self._lane_width + 0.5*self._lane_width)
        left_bound = max(self._ego_lane_index - self._lane_idx, 0)*self._lane_width + 0.5*self._lane_width
        sample_width_d = 0.125*self._lane_width

        di_range = np.arange(right_bound, left_bound, sample_width_d)
        Ti_range = np.arange(MINT, MAXT, DT)
        vi_range = np.array([max(10/3.6, low_resolution_speed)])

        generated_fp, fp_available = self.calculate_path_in_given_range(csp, c_speed, start_state, 
                                                                            di_range, Ti_range, vi_range)

        if fp_available:
            rospy.logdebug("----> Lane_Werling: Successful Planning (Lateral slight adjustment)")
            return generated_fp, low_resolution_speed


        # Speed adjustment
        speed_adjust_thres = 5/3.6
        # speed_low_bound = low_resolution_speed
        # speed_high_bound = max(low_resolution_speed, speed_adjust_thres)
        speed_low_bound = max(low_resolution_speed, 1/3.6)
        speed_high_bound = max(low_resolution_speed, speed_adjust_thres)
        speed_sample_d = 1/3.6 # m/s

        di_range = np.arange(right_bound, left_bound, sample_width_d)
        Ti_range = np.arange(MINT, MAXT, DT)
        vi_range = np.arange(speed_low_bound, speed_high_bound, speed_sample_d)

        generated_fp, fp_available = self.calculate_path_in_given_range(csp, c_speed, start_state, 
                                                                            di_range, Ti_range, vi_range)

        if fp_available:
            rospy.logdebug("----> Lane_Werling: Successful Planning (speed adjustment)")
            return generated_fp, generated_fp.s_d[-1]
            
        rospy.logdebug("----> Lane_Werling: Fail to find a solution")
        return best_free_fp, 0

    def calculate_path_in_given_range(self, csp, c_speed, start_state, di_range, Ti_range, vi_range):

        fplist = self.calc_frenet_paths(c_speed, start_state, di_range, Ti_range, vi_range)
        fplist = self.calc_global_paths(fplist, csp)
        fplist = self.check_paths(fplist)

        if len(fplist) == 0: 
            return None, False

        self.all_trajectory.extend(fplist)

        path_tuples = []
        for fp in fplist:
            one_path = [fp, fp.cf]
            path_tuples.append(one_path)
        
        sorted_fplist = sorted(path_tuples, key=lambda path_tuples: path_tuples[1])
        best_fp, _ = sorted_fplist[0]

        for fp, score in sorted_fplist:
            if self.obs_prediction.check_collision(fp):
                return fp, True

        return best_fp, False

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

    def calc_frenet_paths(self, c_speed, start_state, di_range, Ti_range, vi_range): # input state

        frenet_paths = []

        s0 = start_state.s0
        c_d = start_state.c_d
        c_d_d = start_state.c_d_d
        c_d_dd = start_state.c_d_dd


        for di in di_range:

            # Lateral motion planning
            for Ti in Ti_range:
                fp = Frenet_path()

                lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti) 

                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]                        
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Loongitudinal motion planning
                for tv in vi_range:
                
                    if tv < 0:
                        continue
                
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
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.sqrt(dx**2 + dy**2))

            
            try:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])
            except:
                fp.yaw.append(0.1)
                fp.ds.append(0.1)

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_paths(self, fplist):
        okind = []
        for i, _ in enumerate(fplist):

            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                # rospy.logdebug("exceeding max speed")
                continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                # rospy.logdebug("exceeding max accel")
                continue
            elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
                # rospy.logdebug("exceeding max curvature")
                continue
            # if not self.obs_prediction.check_collision(fplist[i]):
            #     continue

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

        
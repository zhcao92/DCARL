import numpy as np
import bisect
import copy
import time

from Agent.zzz.tools import *
from Agent.zzz.predict import predict
from Agent.zzz.frenet import *
from Agent.zzz.actions import TrajectoryAction




# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 500.0  # maximum curvature [1/m]
D_ROAD_W = 0.6  # road width sampling length [m]
DT = 0.3  # time tick [s]
MAXT = 4.3  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
D_T_S = 5 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed

# collision check
OBSTACLES_CONSIDERED = 5
ROBOT_RADIUS = 2.5  # robot radius [m]
RADIUS_SPEED_RATIO = 0.3 # higher speed, bigger circle
MOVE_GAP = 1.5

# Cost weights
KJ = 0.1 
KT = 0.1 
KD = 1.0
KLAT = 1.0
KLON = 1.0


class LaneTrajectoryPlanner(object):
    def __init__(self):
        self.frenet_lanes = []
        self.last_target_lane_index = -10

    def build_frenet_lane(self, dynamic_map):

        lane_num = len(dynamic_map.lanes)
        for lane_idx, lane in enumerate(dynamic_map.lanes):
            self.frenet_lanes.append(Werling(lane.central_path_array,lane_idx,lane_num))

    def run_step(self, dynamic_map, target_lane_index, desired_speed):
        if dynamic_map.lanes_updated:
            print("[Planning] :  Updated frenets")
            self.frenet_lanes = []
            self.last_target_lane_index = -10
            self.build_frenet_lane(dynamic_map)

        if target_lane_index != self.last_target_lane_index:
            for lane in self.frenet_lanes:
                lane.lane_change_clean_buff()

            self.last_target_lane_index = target_lane_index
        return self.frenet_lanes[int(target_lane_index)].trajectory_update(dynamic_map, desired_speed)


class Werling(object):

    def __init__(self, lane_path, lane_idx, lane_num, lane_width = 3):

        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()

        self.ref_path = None
        self.ref_path_tangets = None
        self.ob = None
        self.csp = None
        self.target_speed = 0

        self._lane_num = lane_num
        self._lane_idx = lane_idx
        self._lane_width = lane_width

        self.build_frenet_path(lane_path)

    def build_frenet_path(self, lane_path):
        if self.csp is None:
            lane_path = [(point.position.x, point.position.y) for point in lane_path]
            lane_path = np.array(lane_path)
            self.ref_path = dense_polyline2d(lane_path, 2)
            self.ref_path_tangets = np.zeros(len(self.ref_path))

            Frenetrefx = self.ref_path[:,0]
            Frenetrefy = self.ref_path[:,1]

            tx, ty, tyaw, tc, self.csp = self.generate_target_course(Frenetrefx,Frenetrefy)
            

    
    def lane_change_clean_buff(self):
        
        self.last_trajectory_array = np.c_[0, 0]
        self.last_trajectory = Frenet_path()
        self.last_trajectory_array_rule = np.c_[0, 0]
        self.last_trajectory_rule = Frenet_path()
    
    def trajectory_update(self, dynamic_map, target_speed):
        self.initialize_obj_predict(dynamic_map)
        self.all_trajectory = []
        self._ego_lane_index = int(round(dynamic_map.ego_vehicle.lane_idx))

        start_state = self.calculate_start_state(dynamic_map)
        # print("[Planning] : Start state",start_state.s0, start_state.c_d)

        generated_trajectory, local_desired_speed = self.frenet_optimal_planning(self.csp, self.c_speed, start_state, target_speed)

        trajectory_array = np.c_[generated_trajectory.x, generated_trajectory.y]
        self.last_trajectory_array_rule = trajectory_array
        self.last_trajectory_rule = generated_trajectory

        trajectory_action = TrajectoryAction(trajectory_array, local_desired_speed)
        # print("[Planning] : Trajectory",trajectory_array)
        return trajectory_action

    def trajectory_update_UBP(self, dynamic_map, UBP_action):
        if UBP_action == 1: # brake!
            generated_trajectory =  self.all_trajectory[0][0]
            trajectory_array = np.c_[generated_trajectory.x, generated_trajectory.y]
            msg = DecisionTrajectory()
            msg.trajectory = convert_ndarray_to_pathmsg(trajectory_array)
            msg.desired_speed = [0] * len(trajectory_array)
            return msg

        fplist = self.all_trajectory      
        bestpath = fplist[int(UBP_action - 2)][0]
        trajectory_array = np.c_[bestpath.x, bestpath.y]
        
        # next time when you calculate start state
        self.last_trajectory_array_rule = trajectory_array
        self.last_trajectory_rule = bestpath 

        trajectory_action = TrajectoryAction(trajectory_array, local_desired_speed)
        print("----> UBP: Successful Planning")           
        return trajectory_action


    def initialize_obj_predict(self, dynamic_map):

        self.obs_prediction = predict(dynamic_map,OBSTACLES_CONSIDERED,MAXT,DT, 
                                        ROBOT_RADIUS,RADIUS_SPEED_RATIO,MOVE_GAP,
                                        dynamic_map.ego_vehicle.v)


    def calculate_start_state(self, dynamic_map):
        start_state = Frenet_state()

        if len(self.last_trajectory_array_rule) > 5:
            # find closest point on the last trajectory
            mindist = float("inf")
            bestpoint = 0
            for t in range(len(self.last_trajectory_rule.x)):
                pointdist = (self.last_trajectory_rule.x[t] - dynamic_map.ego_vehicle.x) ** 2 + (self.last_trajectory_rule.y[t] - dynamic_map.ego_vehicle.y) ** 2
                if mindist >= pointdist:
                    mindist = pointdist
                    bestpoint = t
            start_state.s0 = self.last_trajectory_rule.s[bestpoint]
            start_state.c_d = self.last_trajectory_rule.d[bestpoint]
            start_state.c_d_d = self.last_trajectory_rule.d_d[bestpoint]
            start_state.c_d_dd = self.last_trajectory_rule.d_dd[bestpoint]
            # self.c_speed = self.last_trajectory_rule.s_d[bestpoint]
            self.c_speed = dynamic_map.ego_vehicle.v
        else:
            self.c_speed = dynamic_map.ego_vehicle.v       # current speed [m/s]
            ffstate = get_frenet_state(dynamic_map.ego_vehicle, self.ref_path, self.ref_path_tangets)

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
            print("[Planning] : First try success!")
            return best_free_fp, best_free_fp.s_d[-1]

        # Lateral slight adjustment
        right_bound = - (max((self._lane_idx - self._ego_lane_index),0) * self._lane_width + 0.5 * self._lane_width)
        left_bound = max(self._ego_lane_index - self._lane_idx, 0) * self._lane_width + 0.5 * self._lane_width
        sample_width_d = 0.125 * self._lane_width

        # left_bound = 6.0
        # right_bound = -6.0
        # sample_width_d = 1.0
        # low_resolution_speed = 30 /3.6

        di_range = np.arange(right_bound, left_bound, sample_width_d)
        Ti_range = np.arange(MINT, MAXT, DT)
        vi_range = np.array([max(10/3.6, low_resolution_speed)])

        generated_fp, fp_available = self.calculate_path_in_given_range(csp, c_speed, start_state, 
                                                                            di_range, Ti_range, vi_range)

        if fp_available:
            # print("[Planning] : Second try success!")

            return generated_fp, generated_fp.s_d[-1]

        speed_adjust_thres = 40/3.6
        speed_low_bound = max(low_resolution_speed, 1/3.6)
        speed_high_bound = max(low_resolution_speed, speed_adjust_thres)
        speed_sample_d = 1/3.6 # m/s

        di_range = np.arange(right_bound, left_bound, sample_width_d)
        Ti_range = np.arange(MINT, MAXT, DT)
        vi_range = np.arange(speed_low_bound, speed_high_bound, speed_sample_d)

        generated_fp, fp_available = self.calculate_path_in_given_range(csp, c_speed, start_state, 
                                                                            di_range, Ti_range, vi_range)

        if fp_available:
            # print("[Planning] : Third try success!")

            return generated_fp, generated_fp.s_d[-1]
            
        return best_free_fp, 0

    def calculate_path_in_given_range(self, csp, c_speed, start_state, di_range, Ti_range, vi_range):

        fplist = self.calc_frenet_paths(c_speed, start_state, di_range, Ti_range, vi_range)
        fplist = self.calc_global_paths(fplist, csp)
        # fplist = self.check_paths(fplist)
        # print("fplist",len(fplist))

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
            # print("collision path")
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
                continue
            elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
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



class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)


    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None

        """        

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
 
        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)

        return B


class Spline2D:
    """
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw



import numpy as np
import rospy
import matplotlib.pyplot as plt
import copy
import math

from zzz_planning_decision_continuous_models.Werling import cubic_spline_planner
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from zzz_navigation_msgs.msg import Lane
from zzz_driver_msgs.utils import get_speed
from zzz_cognition_msgs.msg import RoadObstacle
from zzz_common.kinematics import get_frenet_state
from zzz_common.geometry import dense_polyline2d

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 10.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 4.0  # maximum road width [m] # related to RL action space
D_ROAD_W = 2.0  # road width sampling length [m]
DT = 0.15  # time tick [s]
MAXT = 4.6  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
TARGET_SPEED = 15.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 2  # sampling number of target speed
ROBOT_RADIUS = 4.5  # robot radius [m]

# Cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

show_animation = True

class Werling(object):

    def __init__(self):
        self.T = 1.6
        self.g0 = 7
        self.a = 2.73
        self.b = 1.65
        self.delta = 4
        self.decision_dt = 0.75
        self.dynamic_map = None
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

        self.last_trajectory_array = []
        self.last_trajectory = []
        self.last_trajectory_array_rule = []
        self.last_trajectory_rule = []
    
    def update_dynamic_map(self, dynamic_map):
        self.dynamic_map = dynamic_map

    def trajectory_update(self, dynamic_map, closest_obs):
        
        reference_path_from_map = dynamic_map.jmap.reference_path.map_lane.central_path_points
        
        ref_path_ori = self.convert_path_to_ndarray(reference_path_from_map)
        ref_path = dense_polyline2d(ref_path_ori, 1)
        ref_path_tangets = np.zeros(len(ref_path))

        Frenetrefx = ref_path[:,0]
        Frenetrefy = ref_path[:,1]
        now00 = rospy.get_rostime()
        # obstacle lists
        ob = []
        # if dynamic_map.jmap.obstacles is not None:
        #     for obs in dynamic_map.jmap.obstacles:
        #         if abs(obs.state.pose.pose.position.x - dynamic_map.ego_state.pose.pose.position.x) < 50 and abs(obs.state.pose.pose.position.y - dynamic_map.ego_state.pose.pose.position.y) < 50:
        #             obstacle = [obs.state.pose.pose.position.x , obs.state.pose.pose.position.y , obs.state.twist.twist.linear.x , obs.state.twist.twist.linear.y]
        #             ob.append(obstacle)
        for obs in closest_obs:
            ob.append(obs)
        ob = np.array(ob)

        tx, ty, tyaw, tc, csp = generate_target_course(Frenetrefx,Frenetrefy)

        now0 = rospy.get_rostime()
        # print("-----------------------------rule-based time consume inside111",now0.to_sec() - now00.to_sec())
        # initial state
        if len(self.last_trajectory_array_rule) > 5:
            # find closest point on the last trajectory
            mindist = float("inf")
            for t in range(len(self.last_trajectory_rule.t)):
                pointdist = (self.last_trajectory_rule.x[t] - dynamic_map.ego_state.pose.pose.position.x) ** 2 + (self.last_trajectory_rule.y[t] - dynamic_map.ego_state.pose.pose.position.y) ** 2
                if mindist >= pointdist:
                    mindist = pointdist
                    bestpoint = t
            s0 = self.last_trajectory_rule.s[bestpoint]
            c_d = self.last_trajectory_rule.d[bestpoint]
            c_d_d = self.last_trajectory_rule.d_d[bestpoint]
            c_d_dd = self.last_trajectory_rule.d_dd[bestpoint]
            c_speed = self.last_trajectory_rule.s_d[bestpoint]
        else:
            ego_state = dynamic_map.ego_state
            c_speed = get_speed(ego_state)       # current speed [m/s]
            ffstate = get_frenet_state(dynamic_map.ego_state, ref_path, ref_path_tangets)
            c_d = - ffstate.d  # current lateral position [m]
            c_d_d = ffstate.vd  # current lateral speed [m/s]
            c_d_dd = 0   # current latral acceleration [m/s]
            s0 = ffstate.s #+ c_speed * 0.5      # current course position

        now1 = rospy.get_rostime()
        # print("-----------------------------rule-based time consume inside111",now1.to_sec() - now0.to_sec())

        generated_trajectory = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)
        now2 = rospy.get_rostime()
        # print("-----------------------------rule-based time consume inside222",now2.to_sec() - now1.to_sec())


        if generated_trajectory is not None:
            desired_speed = generated_trajectory.s_d[-1] 
            trajectory_array_ori = np.c_[generated_trajectory.x, generated_trajectory.y]
            trajectory_array = dense_polyline2d(trajectory_array_ori,1)
            self.last_trajectory_array_rule = trajectory_array
            self.last_trajectory_rule = generated_trajectory              
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            print("111111111111111111111111111")
            
            # for s in generated_trajectory.s:
                # print("+++++++++++++++++++++++++++++++++++++pathstate.s",s)
        elif len(self.last_trajectory_array_rule) > 5 and c_speed > 1:
            trajectory_array = self.last_trajectory_array_rule
            generated_trajectory = self.last_trajectory_rule
            desired_speed =  0 #generated_trajectory.s_d[-1] 
            print("252525252525252525252525")

        else:
            trajectory_array =  ref_path
            desired_speed = 0
            print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
        # now3 = rospy.get_rostime()
        # print("-----------------------------rule-based time consume inside333",now3.to_sec() - now2.to_sec())
        return trajectory_array, desired_speed, generated_trajectory


    def trajectory_update_withRL_second(self, dynamic_map, RLS_action):
        
        # build frenet
        reference_path_from_map = dynamic_map.jmap.reference_path.map_lane.central_path_points
        ref_path_ori = self.convert_path_to_ndarray(reference_path_from_map)
        ref_path = dense_polyline2d(ref_path_ori, 1)
        ref_path_tangets = np.zeros(len(ref_path))
        Frenetrefx = ref_path[:,0]
        Frenetrefy = ref_path[:,1]
        tx, ty, tyaw, tc, csp = generate_target_course(Frenetrefx,Frenetrefy)

        # obstacle lists
        ob = []
        ob = np.array(ob)

        # emergency stop
        if RLS_action[1] < 3.0/3.6:
            return ref_path, 0.0

        # first trajectory initial state
        if len(self.last_trajectory_array) > 5:
            # find closest point on the last trajectory
            mindist = float("inf")
            for t in range(len(self.last_trajectory.t)):
                pointdist = (self.last_trajectory.x[t] - dynamic_map.ego_state.pose.pose.position.x) ** 2 + (self.last_trajectory.y[t] - dynamic_map.ego_state.pose.pose.position.y) ** 2
                if mindist >= pointdist:
                    mindist = pointdist
                    bestpoint = t
            c_speed = self.last_trajectory.s_d[bestpoint]
            s0 = self.last_trajectory.s[bestpoint] + get_speed(dynamic_map.ego_state) * 1.0
            c_d = self.last_trajectory.d[bestpoint]
            c_d_d = self.last_trajectory.d_d[bestpoint]
            c_d_dd = self.last_trajectory.d_dd[bestpoint]
        else:
            ego_state = dynamic_map.ego_state
            c_speed = get_speed(ego_state)       # current speed [m/s]
            ffstate = get_frenet_state(dynamic_map.ego_state, ref_path, ref_path_tangets)
            c_d = - ffstate.d  # current lateral position [m]
            print("-------------------------- ffstate.d", -ffstate.d)
            print("-------------------------- ffstate.d", -ffstate.d)
            print("-------------------------- ffstate.vd ", ffstate.vd)
            print("-------------------------- ffstate.vd ", ffstate.vd)
            print("-------------------------- ffstate.s", ffstate.s)
            print("-------------------------- ffstate.s", ffstate.s)

            c_d_d = ffstate.vd  # current lateral speed [m/s]
            c_d_dd = 0   # current latral acceleration [m/s]
            s0 = ffstate.s + c_speed * 1.0      # current course position

        first_trajectory = frenet_optimal_planning_withRL(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, RLS_action)
    
        # second trajectory initial state
        s0_s = first_trajectory.s[-1]  #+ c_speed * 0.5      # current course position
        c_d_s = first_trajectory.d[-1]    # current lateral position [m]
        c_d_d_s = 0 #first_trajectory.d_d[-1]  # current lateral speed [m/s]
        c_d_dd_s = 0 #first_trajectory.d_dd[-1]   # current latral acceleration [m/s]
        c_speed_s = first_trajectory.s_d[-1]       # current speed [m/s]
        print("-------------------------- ffstate.s", first_trajectory.s[-1])
        print("-------------------------- ffstate.d", first_trajectory.d[-1])
        print("-------------------------- ffstate.d_d ", first_trajectory.d_d[-1])
        print("-------------------------- ffstate.d_dd", first_trajectory.d_dd[-1])
        print("-------------------------- ffstate.s_d", first_trajectory.s_d[-1])

        generated_trajectory = frenet_optimal_planning(csp, s0_s, c_speed_s, c_d_s, c_d_d_s, c_d_dd_s, ob)

        if generated_trajectory is not None:
            connect_x = np.r_[first_trajectory.x, generated_trajectory.x]
            connect_y = np.r_[first_trajectory.y, generated_trajectory.y]
            trajectory_array_ori = np.c_[connect_x, connect_y]          
            trajectory_array = dense_polyline2d(trajectory_array_ori,1)
            desired_speed = RLS_action[1] #first_trajectory.s_d[-1] 
            self.last_trajectory_array_rule = trajectory_array
            self.last_trajectory_rule = generated_trajectory       

            print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")
            # print("111111111111111111111111111")


        else:
            trajectory_array_ori = np.c_[first_trajectory.x, first_trajectory.y]
            trajectory_array = dense_polyline2d(trajectory_array_ori, 1)
            desired_speed = RLS_action[1] #first_trajectory.s_d[-1] 

            print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")
            # print("3333333333333333333333333333")

        return trajectory_array, desired_speed

    def convert_path_to_ndarray(self, path):
        point_list = [(point.position.x, point.position.y) for point in path]
        return np.array(point_list)


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    now00 = rospy.get_rostime()

    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    now0 = rospy.get_rostime()
    print("-----------------------------frenet time consume inside111",now0.to_sec() - now00.to_sec())
    print("-----------------------------fplist",len(fplist))
    fplist = calc_global_paths(fplist, csp)
    now1 = rospy.get_rostime()
    print("-----------------------------frenet time consume inside111",now1.to_sec() - now0.to_sec())
    print("-----------------------------fplist",len(fplist))

    fplist = check_paths(fplist, ob)
    now2 = rospy.get_rostime()
    print("-----------------------------frenet time consume inside111",now2.to_sec() - now1.to_sec())
    print("-----------------------------fplist",len(fplist))

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp
    return bestpath


def frenet_optimal_planning_withRL(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, RLS_action):
    # this function use rl point to find the closest trajectory with the rl point at 0.75s
    
    Ti = 2.25
    tv = RLS_action[1]

    if RLS_action[1] < 5/3.6:
        di = 0.0
    else:
        di = RLS_action[0]



    # calculate the only trajectory
    fplist = []
    fp = Frenet_path()
    lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti) 
    fp.t = [t for t in np.arange(0.0, Ti, DT)]
    fp.d = [lat_qp.calc_point(t) for t in fp.t]                        
    fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
    fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
    fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

    tfp = copy.deepcopy(fp)
    lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

    # square of diff from target speed
    ds = (TARGET_SPEED - tfp.s_d[-1])**2
    tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
    tfp.cv = KJ * Js + KT * Ti + KD * ds
    tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

    fplist.append(tfp)    
    
    # convert to global
    fplist = calc_global_paths(fplist, csp)

    bestpath = fplist[0]

    return bestpath


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0): # input state

    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):# sampling -7-7   1

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):#4 5  0.2
            fp = Frenet_path()

            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti) 

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]                        
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Loongitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1])**2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):

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


def prediction_obstacle(ob, prediction_model): # we should do prediciton in driving space
    
    obs_paths = []

    if prediction_model == 1:
        for one_ob in ob:
            obsp = Frenet_path()
            obsp.t = [t for t in np.arange(0.0, MAXT, DT)]
            for i in range(len(obsp.t)):
                obspx = one_ob[0] + i * DT * one_ob[2]
                obspy = one_ob[1] + i * DT * one_ob[3]
                obsp.x.append(obspx)
                obsp.y.append(obspy)
                # obsp.x = [x for x in np.arange(one_ob[0] , one_ob[0] + MAXT * one_ob[2] ,DT * one_ob[2])] # keep speed
                # obsp.y = [y for y in np.arange(one_ob[1] , one_ob[1] + MAXT * one_ob[3] ,DT * one_ob[3])] # keep speed
            obs_paths.append(obsp)

    return obs_paths

def check_collision(fp, ob):

    obs_paths = prediction_obstacle(ob,1)
    if len(obs_paths) == 0:
        return True
    try:
        for obsp in obs_paths:
            for t in range(len(fp.t)):
                d = (obsp.x[t] - fp.x[t])**2 + (obsp.y[t] - fp.y[t])**2
                if d <= ROBOT_RADIUS**2 and t > 1: # collision , t>0.75 to avoid to check the ego vehicle state
                    return False
    except:
        pass

    return True


    # if len(ob) == 0:
    #     return True

    # for i in range(len(ob[:, 0])):
    #     d = [((ix - ob[i, 0])**2 + (iy - ob[i, 1])**2)
    #         for (ix, iy) in zip(fp.x, fp.y)]

    #     collision = any([di <= ROBOT_RADIUS**2 for di in d])

    #     if collision:
    #         return False

    # return True


def check_paths(fplist, ob):

    okind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        if not check_collision(fplist[i], ob):
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
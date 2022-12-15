
import numpy as np
import numpy.linalg as npl
import math


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

def wrap_angle(theta):
    '''
    Normalize the angle to [-pi, pi]

    :param float theta: angle to be wrapped
    :return: wrapped angle
    :rtype: float
    '''

    return (theta + np.pi) % (2*np.pi) - np.pi

def polyline_length(line):
    '''
    line: np.array
    '''
    
    if len(line) <= 1:
        return 0
    
    dist_list = np.cumsum(np.linalg.norm(np.diff(line,axis = 0),axis = 1))

    return dist_list[-1]


def dense_polyline2d(line, resolution):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """

    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0],s])
    num = int(round(s[-1]/resolution))

    try:
        s_space = np.linspace(0,s[-1],num = num)
    except:
        raise ValueError(num, s[-1], len(s))


    x = np.interp(s_space,s,line[:,0])
    y = np.interp(s_space,s,line[:,1])

    return np.array([x,y]).T

def dense_polyline2d_withvelocity(line, velocity, resolution, interp="linear"):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """
    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    if interp != "linear":
        raise NotImplementedError("Other interpolation method is not implemented!")

    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0],s])
    num = math.ceil(s[-1]/resolution)

    s_space = np.linspace(0,s[-1],num = num)
    x = np.interp(s_space,s,line[:,0])
    y = np.interp(s_space,s,line[:,1])
    waypoints_velocity = np.interp(s_space,s,velocity)

    return np.array([x,y]).T, waypoints_velocity


def dist_from_point_to_line2d(x0,y0,x1,y1,x2,y2):

    l = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
    
    # if two points are the same one
    if l == 0:
       dl = math.sqrt((y0-y1)*(y0-y1) + (x0-x1)*(x0-x1))
       return dl, 0, 0

    dl = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - x1*y2) / l
    d1 = (x1*x1+x0*(x2-x1)-x1*x2 + y1*y1+y0*(y2-y1)-y1*y2) / l
    d2 = (x2*x2-x0*(x2-x1)-x1*x2 + y2*y2-y0*(y2-y1)-y1*y2) / l
    
    return dl, d1, d2


def dist_from_point_to_polyline2d(x0,y0,line,return_end_distance = False):

    if len(line) < 2:
        raise ValueError("Cannot calculate distance to an empty line or a single point!")

    dist_line = npl.norm(line - [x0, y0], axis=1) # dist from current point (x0, y0) to line points
    closest_idx = np.argmin(dist_line) # index of closet point

    length = len(line)
    closest_type = 0 # 0: closest point, 1: next line of closest point, -1: previous line of closest point

    # Calculate distances mentioned above
    if closest_idx == 0: # When this point is at line start
        dist_next, dist_next_head, dist_next_tail = dist_from_point_to_line2d(x0, y0,
            line[0,0], line[0,1], line[1,0], line[1,1])
        if dist_next_head < 0: # case 1
            dist_closest = dist_line[closest_idx] # out of the start of the lane to get an obtuse angle
            #jxy: add sign here
            if dist_next < 0:
                dist_closest *= -1
        else: # case 2
            dist_closest = dist_next
            closest_type = 1
    elif closest_idx == length - 1: # When this point is at end
        dist_previous, dist_previous_head, dist_previous_tail = dist_from_point_to_line2d(x0, y0,
            line[length-2, 0], line[length-2, 1], line[length-1, 0], line[length-1, 1])
        if dist_previous_tail < 0: # case 3
            dist_closest = dist_line[closest_idx] # out of the end of the lane to get an obtuse angle
            #jxy: add sign here
            if dist_previous < 0:
                dist_closest *= -1
        else: # case 4
            dist_closest = dist_previous
            closest_type = -1
    else:
        dist_previous, dist_previous_head, dist_previous_tail = dist_from_point_to_line2d(x0, y0,
            line[closest_idx-1, 0], line[closest_idx-1, 1],
            line[closest_idx, 0], line[closest_idx, 1])
        dist_next, dist_next_head, dist_next_tail = dist_from_point_to_line2d(x0, y0,
            line[closest_idx, 0], line[closest_idx, 1],
            line[closest_idx+1, 0], line[closest_idx+1, 1])
        if dist_previous_tail < 0 and dist_next_head < 0: # case 5
            dist_closest = dist_line[closest_idx]
            # sign determination
            if dist_from_point_to_line2d(
                line[closest_idx+1, 0], line[closest_idx+1, 1],
                line[closest_idx-1, 0], line[closest_idx-1, 1],
                line[closest_idx, 0], line[closest_idx, 1])[0] > 0:
                dist_closest *= -1
        elif dist_previous_tail < 0: # case 6
            dist_closest = dist_next
            closest_type = 1
        elif dist_next_head < 0: # case 7
            dist_closest = dist_previous
            closest_type = -1
        else: # case 8
            if abs(dist_next) > abs(dist_previous):
                dist_closest = dist_previous
                closest_type = -1
            else:
                dist_closest = dist_next
                closest_type = 1

    # Return early if total distance is not needed
    if not return_end_distance:
        return (dist_closest, closest_idx, closest_type)

    # Calculate accumulated distance
    if closest_type == 1:
        # closer to next line segment
        dist_start = dist_next_head + np.sum(npl.norm(np.diff(line[:closest_idx+1], axis=0), axis=1))
        dist_end = dist_next_tail + np.sum(npl.norm(np.diff(line[closest_idx+1:], axis=0), axis=1))
    elif closest_type == -1:
        # closer to previous line segment
        dist_start = dist_previous_head + np.sum(npl.norm(np.diff(line[:closest_idx], axis=0), axis=1))
        dist_end = dist_previous_tail + np.sum(npl.norm(np.diff(line[closest_idx:], axis=0), axis=1))
    else:
        # closer to current point
        dist_start = np.sum(npl.norm(np.diff(line[:closest_idx+1], axis=0), axis=1))
        dist_end = np.sum(npl.norm(np.diff(line[closest_idx:], axis=0), axis=1))

    return dist_closest, closest_idx, closest_type, dist_start, dist_end

def get_frenet_state(vehicle,polyline,tangents):

    dist, nearest_idx, nearest_type, dist_start, dist_end = dist_from_point_to_polyline2d(
        vehicle.x,vehicle.y,polyline, return_end_distance=True)

    if nearest_type == 1:
        psi = math.atan2(
            polyline[nearest_idx+1, 1] - polyline[nearest_idx, 1],
            polyline[nearest_idx+1, 0] - polyline[nearest_idx, 0])
    elif nearest_type == -1:
        psi = math.atan2(
            polyline[nearest_idx, 1] - polyline[nearest_idx-1, 1],
            polyline[nearest_idx, 0] - polyline[nearest_idx-1, 0],
        )
    else:
        psi = tangents[nearest_idx]

    rot = np.array([
        [math.cos(psi), math.sin(psi)],
        [-math.sin(psi), math.cos(psi)]
    ])

    yaw = vehicle.yaw

    frenet = Frenet_state()
    frenet.s = dist_start
    frenet.d = dist
    frenet.psi = wrap_angle(yaw - psi) 

    v = np.array([vehicle.vx, vehicle.vy])
        
    frenet.vs, frenet.vd = ((v.T).dot(rot.T)).T
    
    return frenet

def convert_path_to_ndarray(path):
    point_list = [(point.position.x, point.position.y) for point in path]
    return np.array(point_list)

def pointcurvature(x,y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """
    t_a = npl.norm([x[1]-x[0],y[1]-y[0]])
    t_b = npl.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(npl.inv(M),x)
    b = np.matmul(npl.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

def linecurvature(line):
    """
    input  : the pololines (np.array)
    output : the curvature of the lines
    """
    
    ka = []
    ka.append(0)
    no = []
    
    for idx in range(len(line)-2):
        x = np.array([line[idx][0], line[idx+1][0], line[idx+2][0]])
        y = np.array([line[idx][1], line[idx+1][1], line[idx+2][1]])
        kappa, norm = pointcurvature(x,y)
        ka.append(kappa)
        no.append(norm)
    
    ka.append(ka[-1])

    return np.array(ka)

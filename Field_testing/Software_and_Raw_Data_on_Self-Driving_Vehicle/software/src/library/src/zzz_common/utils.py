import math
import numpy as np


# todo write this in cython and move them to geometry
def quaternion_to_eular(x, y, z, w):

    roll = math.atan2(2 * (w * x + y * z),
                      1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y),
                     1 - 2 * (y * y + z * z))

    return roll, pitch, yaw

def calc_vehicle_corners(v_x, v_y, length, width, yaw):

    rotation_mat = np.array([[math.cos(yaw), - math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    # FL, FR, RL, RR
    relative_corner = np.array([[length / 2, length / 2, - length / 2, - length / 2],
                                   [width / 2, - width / 2, width / 2, - width / 2]])
    relative_rotated_corner = np.matmul(rotation_mat, relative_corner)
    vehicle_position = np.array([[v_x, v_x, v_x, v_x],
                              [v_y, v_y, v_y, v_y]])
    real_corner = vehicle_position + relative_rotated_corner
    return real_corner
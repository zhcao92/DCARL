'''
This package contains useful message conversions
'''

import numpy as np
from zzz_navigation_msgs.msg import Map

def default_msg(msg_type):
    '''
    Setting default values for the messages
    '''
    if msg_type == Map:
        msg = Map()
        msg.in_junction = True
    else:
        raise ValueError("Unrecognized message type")
    
    return msg

def get_lane_array(lanes):
    arrays = []
    for lane in lanes:
        point_list = [(point.position.x, point.position.y) for point in lane.central_path_points]
        arrays.append(np.array(point_list))
    return arrays

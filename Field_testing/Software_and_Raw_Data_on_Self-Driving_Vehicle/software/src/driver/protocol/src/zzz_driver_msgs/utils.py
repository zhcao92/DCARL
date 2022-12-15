import math
import tf.transformations as tft

from zzz_driver_msgs.msg import RigidBodyState, RigidBodyStateStamped
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance

def get_speed(msg):
    '''
    Calculate speed value
    '''
    if type(msg) == Twist:
        return math.sqrt(msg.linear.x**2 + msg.linear.y**2 + msg.linear.z**2)
    elif type(msg) == TwistWithCovariance or type(msg) == RigidBodyState:
        return get_speed(msg.twist)
    elif type(msg) == RigidBodyStateStamped:
        return get_speed(msg.state)
    else:
        raise ValueError("Incorrect message type for get_speed")

def get_yaw(msg):
    '''
    Calculate yaw angle assuming on 2d plane
    '''
    if type(msg) == Pose:
        _,_,yaw = tft.euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        return yaw
    elif type(msg) == PoseWithCovariance or type(msg) == RigidBodyState:
        return get_yaw(msg.pose)
    elif type(msg) == RigidBodyStateStamped:
        return get_yaw(msg.state)
    else:
        raise ValueError("Incorrect message type for get_yaw")

def convert_odometry_to_state(msg):
    '''
    Convert nav_msgs/Odometry to zzz_driver_msgs/RigidBodyStateStamped
    '''

    new_msg = RigidBodyStateStamped()
    new_msg.header = msg.header
    new_msg.state.child_frame_id = msg.child_frame_id

    new_msg.state.pose = msg.pose
    new_msg.state.twist = msg.twist
    return new_msg

from zzz_common.kinematics import get_absolute_state
from zzz_driver_msgs.msg import RigidBodyStateStamped
from zzz_cognition_msgs.msg import LaneState, MapState, RoadObstacle, JunctionMapState
from zzz_perception_msgs.msg import TrackingBoxArray, ObjectClass, DimensionWithCovariance
import numpy as np

def default_msg(msg_type):
    '''
    Setting default values for the messages
    '''
    if msg_type == LaneState:
        msg = LaneState()
        msg.stop_distance = float('inf')
    elif msg_type == MapState:
        msg = MapState()
    elif msg_type == RoadObstacle:
        msg = RoadObstacle()
    elif msg_type == JunctionMapState:
        msg = JunctionMapState()
    else:
        raise ValueError("Unrecognized message type")
    
    return msg

def convert_tracking_box(array, pose):
    '''
    Convert tracking box into RoadObstacle. Pose should be RigidBodyStateStamped which is got from /zzz/navigation/ego_pose
    '''
    assert type(array) == TrackingBoxArray
    assert type(pose) == RigidBodyStateStamped

    obstacles = []
    
    for obj in array.targets:
        if array.header.frame_id != pose.header.frame_id:
            trackpose = RigidBodyStateStamped()
            trackpose.header = array.header
            trackpose.state.pose = obj.bbox.pose
            trackpose.state.twist = obj.twist
            trackpose.state.accel = obj.accel
            # abspose = get_absolute_state(trackpose, pose)
            # TODO
            abspose = trackpose
            assert abspose.header.frame_id == 'map'
        else:
            abspose = RigidBodyStateStamped()
            abspose.header = array.header
            abspose.state.pose = obj.bbox.pose
            abspose.state.twist = obj.twist
            abspose.state.accel = obj.accel

        obstacle = RoadObstacle()
        obstacle.uid = obj.uid
        obstacle.state = abspose.state
        if len(obj.classes) > 0:
            obstacle.cls = obj.classes[0]
            if obstacle.cls.classid == 0:
                continue
        else:
            obstacle.cls.classid = ObjectClass.UNKNOWN
            obstacle.cls.score = 1
        # Convert obstacle shape
        obstacle.dimension = obj.bbox.dimension

        #print("twist before transform: %f %f %f\n", obstacle.state.twist.twist.linear.x, obstacle.state.twist.twist.linear.y, obstacle.state.twist.twist.linear.z)
        
        # jxy: Convert velocity (twist)
        # x = obj.bbox.pose.pose.orientation.x
        # y = obj.bbox.pose.pose.orientation.y
        # z = obj.bbox.pose.pose.orientation.z
        # w = obj.bbox.pose.pose.orientation.w

        # rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
        # rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

        # vel_self = np.array([[obj.twist.twist.linear.x], [obj.twist.twist.linear.y], [obj.twist.twist.linear.z]])
        # vel_world = np.matmul(rotation_mat_inverse, vel_self)
        
        #print("shape: ", vel_self.shape, vel_world.shape)
        #check if it should be reversed
        obstacle.state.twist.twist.linear.x = obj.twist.twist.linear.x
        obstacle.state.twist.twist.linear.y = obj.twist.twist.linear.y
        obstacle.state.twist.twist.linear.z = obj.twist.twist.linear.z

        #print("quaternion: %f %f %f %f\n", x, y, z, w)

        #print("transformed twist: %f %f %f\n", obstacle.state.twist.twist.linear.x, obstacle.state.twist.twist.linear.y, obstacle.state.twist.twist.linear.z)

        obstacles.append(obstacle)

    return obstacles

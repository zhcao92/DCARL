
import numpy as np
import rospy

from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from zzz_planning_msgs.msg import DecisionTrajectory
from common import rviz_display, convert_ndarray_to_pathmsg, convert_path_to_ndarray



class Follow_Ref_Path(object):

    def __init__(self):
       pass

    def trajectory_update(self, dynamic_map):
        reference_path_from_map = dynamic_map.jmap.reference_path
        desired_speed = 5 # m/s

        # Process reference path
        if reference_path_from_map is not None:
            send_trajectory = self.process_reference_path(reference_path_from_map, dynamic_map, desired_speed)

            # Convert to Message type
            if send_trajectory is not None:
                msg = DecisionTrajectory()
                msg.trajectory = convert_ndarray_to_pathmsg(send_trajectory) # TODO: move to library
                msg.desired_speed = desired_speed 
                return msg

        else:
            return None

    def process_reference_path(self,trajectory,dynamic_map,desired_speed):
        resolution=0.5
        time_ahead=5
        distance_ahead=10
        rectify_thres=2
        lc_dt = 1.5
        lc_v = 2.67
        ego_x = dynamic_map.ego_state.pose.pose.position.x
        ego_y = dynamic_map.ego_state.pose.pose.position.y
        central_path = convert_path_to_ndarray(trajectory.map_lane.central_path_points)
        dense_centrol_path = dense_polyline2d(central_path, resolution)
        nearest_dis, nearest_idx, _ = dist_from_point_to_polyline2d(ego_x, ego_y, dense_centrol_path)
        nearest_dis = abs(nearest_dis)
        front_path = dense_centrol_path[nearest_idx:]
        dis_to_ego = np.cumsum(np.linalg.norm(np.diff(front_path, axis=0), axis = 1))
        return front_path[:np.searchsorted(dis_to_ego, desired_speed*time_ahead+distance_ahead)-1]   


    def clear_buff(self, dynamic_map):
        pass

    
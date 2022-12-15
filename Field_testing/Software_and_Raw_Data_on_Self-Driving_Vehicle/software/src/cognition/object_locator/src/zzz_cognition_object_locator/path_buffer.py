
import rospy
import numpy as np
from collections import deque
from easydict import EasyDict as edict

from zzz_cognition_msgs.msg import MapState
from zzz_cognition_msgs.utils import default_msg
from zzz_driver_msgs.msg import RigidBodyStateStamped
from zzz_navigation_msgs.msg import LanePoint
from zzz_common.geometry import dense_polyline2d, dist_from_point_to_polyline2d
from nav_msgs.msg import Path
from zzz_driver_msgs.utils import get_speed
from geometry_msgs.msg import PoseStamped
import copy
from threading import Lock

class PathBuffer:
    def __init__(self, buffer_size=150):
        self._buffer_size = buffer_size

        # Buffer for updating states
        self._static_map_buffer = None
        self._static_map_lock = Lock()

        self._ego_vehicle_state_buffer = None
        self._ego_vehicle_state_lock = Lock()

        self._reference_path_buffer = None
        self._reference_path_received = None
        self._reference_path_lock = Lock()

        self._reference_path_segment = deque(maxlen=buffer_size)
        self._reference_path_changed = False # Use this flag to avoid race condition on self._reference_path_segment
        self.ref_path_msg = None

        self._judge_lane_change_threshold = 3
        self._reference_path = None

        self._rerouting_trigger = None
        self._rerouting_sent = False
        self._debug = True

    def set_rerouting_trigger(self, trigger):
        self._rerouting_trigger = trigger

    def receive_static_map(self, map_input):
        assert type(map_input) == MapState
        with self._static_map_lock:
            self._static_map_buffer = map_input
        rospy.logdebug("Updating local dynamic map")

    def receive_ego_state(self, state):
        assert type(state) == RigidBodyStateStamped
        with self._ego_vehicle_state_lock:
            self._ego_vehicle_state_buffer = state

    def receive_reference_path(self, reference_path):
        # TODO: Define a custom reference_path?
        assert type(reference_path) == Path
        if self._debug:
            print("#### PathBuffer msg frame - {}".format(reference_path.header.frame_id))

        # Here reference path is appended reversely in order to easy move points in a FIFO way
        with self._reference_path_lock:
            _reference_path_received = [(waypoint.pose.position.x, waypoint.pose.position.y) 
                                        for waypoint in reference_path.poses]
            self._reference_path_changed = True
            self._reference_path_received = dense_polyline2d(np.array(_reference_path_received),2)
        rospy.loginfo("Received reference path, length:%d", len(reference_path.poses))

    def _relocate_dense_reference_path_buffer(self, ego_state, resolution = 2):
        
        _reference_path_buffer_t = self._reference_path_received
        _, nearest_idx, _ = dist_from_point_to_polyline2d(
                ego_state.pose.pose.position.x,
                ego_state.pose.pose.position.y,
                _reference_path_buffer_t
            )
        self._reference_path_buffer = copy.deepcopy(_reference_path_buffer_t[max(0,nearest_idx-100):]).tolist()

    def update(self, required_reference_path_length = 15,
                prepare_stop_path_length = 30,
                remained_passed_point = 5):
        """
        Delete the passed point and add more point to the reference path
        """
        # Load states
        # if not self._reference_path_buffer or not self._ego_vehicle_state_buffer:
        if not self._ego_vehicle_state_buffer:
            return None

        tstates = edict()
        # clone dynamic_map
        with self._static_map_lock:
            tstates.dynamic_map = copy.deepcopy(self._static_map_buffer or default_msg(MapState)) 

        # clone ego state
        with self._ego_vehicle_state_lock:
            tstates.ego_state = copy.deepcopy(self._ego_vehicle_state_buffer.state)
        
        dynamic_map = tstates.dynamic_map # for easy access
        ego_state = tstates.ego_state # for easy access

        if self._reference_path_received is None:
            return None

        # Process segment clear request
        if self._reference_path_changed:
            self._reference_path_segment.clear()
            self._reference_path_changed = False
            self._relocate_dense_reference_path_buffer(ego_state)
        # tstates.reference_path = self._reference_path_buffer
        # reference_path = tstates.reference_path # for easy access

        # Remove passed waypoints - dequeue
        if len(self._reference_path_segment) > 1:
            _, nearest_idx, _ = dist_from_point_to_polyline2d(
                ego_state.pose.pose.position.x,
                ego_state.pose.pose.position.y,
                np.array(self._reference_path_segment)
            )

            for _ in range(nearest_idx-remained_passed_point):
                removed_point = self._reference_path_segment.popleft()
                rospy.logdebug("removed waypoint: %s, remaining count: %d", str(removed_point), len(self._reference_path_buffer))

        # Choose points from reference path to buffer - enqueue
        while self._reference_path_buffer and len(self._reference_path_segment) < self._buffer_size:
            wp = self._reference_path_buffer.pop(0) # notice that the points are inserted reversely
            self._reference_path_segment.append(wp)

        self.draw_ref_path()

        # Put buffer into dynamic map
        for wp in self._reference_path_segment:
            point = LanePoint()
            point.position.x = wp[0]
            point.position.y = wp[1]
            dynamic_map.jmap.reference_path.map_lane.central_path_points.append(point)
        
        dynamic_map.jmap.reference_path.map_lane.index = -1

        # Reference Path Tail Part:
        # Current reference path is too short, require a new reference path
        ego_v = get_speed(dynamic_map.ego_state)
        if (len(self._reference_path_buffer)+len(self._reference_path_segment) < required_reference_path_length) and ego_v < 1/3.6:
            self.renew_ref_path() # should get from navigation module
        
        if len(self._reference_path_buffer)+len(self._reference_path_segment) < prepare_stop_path_length:
            dynamic_map.model = MapState.MODEL_JUNCTION_MAP

        # TODO: read or detect speed limit
        dynamic_map.jmap.reference_path.map_lane.speed_limit = 30
        return dynamic_map

    def draw_ref_path(self):

        self.ref_path_msg = Path()
        self.ref_path_msg.header.frame_id = "map"

        for wp in self._reference_path_segment:
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            self.ref_path_msg.poses.append(pose)

    def renew_ref_path(self):

        _reference_path_buffer_t = self._reference_path_received
        self._reference_path_buffer = copy.deepcopy(_reference_path_buffer_t.tolist())
        self._reference_path_segment.clear()


    # Current reference path is too short, require a new reference path
        # if len(reference_path) < required_reference_path_length
        #     if not self._rerouting_trigger:
        #         self._rerouting_trigger()
        #         self._rerouting_sent = True
        # else:
        #     self._rerouting_sent = False # reset flag

import socket
import msgpack
import rospy
import numpy as np


from zzz_cognition_msgs.msg import MapState
from zzz_planning_decision_lane_models.longitudinal import IDM
from zzz_planning_decision_lane_models.lateral import LaneUtility
from zzz_driver_msgs.utils import get_speed
from zzz_common.geometry import dense_polyline2d
from zzz_common.kinematics import get_frenet_state



class RLSDecision(object):
    """
    Parameter:
        mode: ZZZ TCP connection mode (client/server)
    """
    def __init__(self, openai_server="127.0.0.1", port=2345, mode="client", recv_buffer=4096):
        self._dynamic_map = None
        self._socket_connected = False
        self._rule_based_longitudinal_model_instance = IDM()
        self._rule_based_lateral_model_instance = LaneUtility(self._rule_based_longitudinal_model_instance)
        self._buffer_size = recv_buffer
        self.collision_signal = False
        self.collision_times = 0

        self.inside_lane = None
        self.outside_lane = None

        self._out_multilane = True
        self.decision_action = 0


        if mode == "client":
            rospy.loginfo("Connecting to RL server...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((openai_server, port))
            self._socket_connected = True
            rospy.loginfo("Connected...")
        else:
            # TODO: Implement server mode to make multiple connection to this node.
            #     In this mode, only rule based action is returned to system
            raise NotImplementedError("Server mode is still wating to be implemented.")

    def enter_junction(self):
        '''
        Send done to AI
        '''
        self._out_multilane = True
        collision = int(self.collision_signal)
        self.collision_signal = False
        leave_current_mmap = 1
        sent_RL_msg = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        sent_RL_msg.append(collision)
        sent_RL_msg.append(leave_current_mmap)
        self.sock.sendall(msgpack.packb(sent_RL_msg))

        try:
            RLS_action = msgpack.unpackb(self.sock.recv(self._buffer_size))
        except:
            pass

    def lateral_decision(self, dynamic_map):

        self._dynamic_map = dynamic_map
        self._rule_based_longitudinal_model_instance.update_dynamic_map(dynamic_map)

        self._out_multilane = False
        RL_state = self.wrap_state()
        sent_RL_msg = RL_state

        collision = int(self.collision_signal)
        self.collision_signal = False
        leave_current_mmap = 0
        sent_RL_msg.append(collision)
        sent_RL_msg.append(leave_current_mmap)

        try:
            self.sock.sendall(msgpack.packb(sent_RL_msg))
            RLS_action = msgpack.unpackb(self.sock.recv(self._buffer_size))
            RLS_action = RLS_action
            print("received action:", RLS_action)
            return self.get_decision_from_discrete_action(RLS_action)
        except:
            return self.get_decision_from_discrete_action(0)

    def wrap_state(self):     

        state = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        state[0] = 0
        state[1] = self._dynamic_map.mmap.ego_lane_index
        state[2] = get_speed(self._dynamic_map.ego_state)
        state[3] = self._dynamic_map.ego_ffstate.vd

        for k, lane in enumerate(self._dynamic_map.mmap.lanes):

            if len(lane.front_vehicles) > 0:
                fv = lane.front_vehicles[0]
                fv_id = fv.uid
                fv_s = fv.ffstate.s
                fv_d = fv.lane_index
                fv_vs = fv.ffstate.vs
                fv_vd = fv.ffstate.vd
            else:
                fv_s = 50
                fv_d = k
                fv_vs = 20
                fv_vd = 0
            
            state[k*4+4] = fv_s
            state[k*4+5] = fv_d
            state[k*4+6] = fv_vs
            state[k*4+7] = fv_vd
        
        if len(self._dynamic_map.mmap.lanes) == 1:
            state[8] = 50
            state[9] = 1
            state[10] = 20
            state[11] = 0

        for k, lane in enumerate(self._dynamic_map.mmap.lanes):
            
            if len(lane.rear_vehicles) > 0:
                rv = lane.rear_vehicles[0]
                rv_id = rv.uid
                rv_s = rv.ffstate.s
                rv_d = rv.lane_index
                rv_vs = rv.ffstate.vs
                rv_vd = rv.ffstate.vd
            else:
                rv_s = -50
                rv_d = k
                rv_vs = 0
                rv_vd = 0
            
            state[k*4+12] = rv_s
            state[k*4+13] = rv_d
            state[k*4+14] = rv_vs
            state[k*4+15] = rv_vd

        if len(self._dynamic_map.mmap.lanes) == 1:
            state[16] = -50 
            state[17] = 1
            state[18] = 0
            state[19] = 0
        return state

    def RL_model_matching(self):
        pass

    def get_decision_from_discrete_action(self, action, acc = 2, decision_dt = 0.75, hard_brake = 4):

        self.decision_action = action

        if len(self._dynamic_map.mmap.lanes) == 1:
            self.inside_lane = 0
            self.outside_lane = 0
        else:
            self.inside_lane = 1
            self.outside_lane = 0
        
        # Rule-based action
        if action == 0:
            print("+++++++++++++++++++")
            self.decision_action = 0.0
            return self._rule_based_lateral_model_instance.lateral_decision(self._dynamic_map)

        current_speed = get_speed(self._dynamic_map.ego_state)
        ego_y = int(round(self._dynamic_map.mmap.ego_lane_index))

        # Hard-brake action
        if action == 1:
            self.decision_action = 1.0
            return ego_y, current_speed - hard_brake * decision_dt

        # ego lane action
        if action == 2:
            self.decision_action = 2.0
            return self.outside_lane, current_speed

        if action == 3:
            self.decision_action = 3.0
            return self.inside_lane, current_speed

        if action == 4:
            self.decision_action = 4.0
            return self.outside_lane, current_speed + acc * decision_dt

        if action == 5:
            self.decision_action = 5.0
            return self.inside_lane, current_speed + acc * decision_dt

        if action == 6:
            self.decision_action = 6.0
            return self.outside_lane, current_speed - acc * decision_dt

        if action == 7:
            self.decision_action = 7.0
            return self.inside_lane, current_speed - acc * decision_dt

        print("------------------------Wrong action type")
        self.decision_action = -1.0
        return self._rule_based_lateral_model_instance.lateral_decision(self._dynamic_map)

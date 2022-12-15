from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import socket
import sys
import time
import weakref

import matplotlib.pyplot as plt
import msgpack
import networkx as nx
import numpy as np

import gym
from gym import core, error, spaces, utils
from gym.utils import seeding

# from carla import Location, Rotation, Transform

##########################################

class ZZZCarlaEnv(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, zzz_client="127.0.0.1", port=2333, recv_buffer=4096, socket_time_out = 1000):
    
        self._restart_motivation = 0
        self.state = []
        self.steps = 1
        self.collision_times = 0

        # Socket
        socket.setdefaulttimeout(socket_time_out) # Set time out
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(socket_time_out) # Set time out
        self.sock.bind((zzz_client, port))
        self.sock.listen()
        self.sock_conn = None
        self.sock_buffer = recv_buffer
        self.sock_conn, addr = self.sock.accept()
        self.sock_conn.settimeout(socket_time_out) # Set time out
        self.rule_based_action = []
        print("ZZZ connected at {}".format(addr))

        # Set action space
        low_action = np.array([-2.0,-15/3.6]) # di - ROAD_WIDTH, tv - TARGET_SPEED - D_T_S * N_S_SAMPLE
        high_action = np.array([2.0, 15/3.6])  #Should be symmetry for DDPG
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.state_dimention = 16

        low  = np.array([-100,  -100,   -20,  -20,  -100, -100,  -20,   -20,   -100, -100,   -20,  -20, -100,  -100, -20, -20])
        high = np.array([100, 100, 20, 20, 100, 100, 20, 20, 100, 100, 20, 20,100, 100, 20, 20])    

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()


    def step(self, action, q_value, rule_action, rule_q):

        # send action to zzz planning module
        
        action = action.astype(float)
        action = action.tolist()
        print("-------------",type(action),action)
        while True:
            try:
                send_action = action
                send_action.append(q_value)
                send_action.append(rule_q)                
                self.sock_conn.sendall(msgpack.packb(send_action))

                # wait next state
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in step")
                self.state = received_msg[0:16]
                collision = received_msg[16]
                leave_current_mmap = received_msg[17]
                threshold = received_msg[18]
                RLpointx = received_msg[19]
                RLpointy = received_msg[20]
                self.rule_based_action = [(RLpointx, RLpointy)]

                # calculate reward
                reward = 5 - (abs(action[0] - RLpointx) + abs(action[1] - RLpointy)) + 0.5 * received_msg[0]
              
                # judge if finish
                done = False

                if collision:
                    done = True
                    #reward = 0#-1000
                    # print("+++++++++++++++++++++ received collision")
                
                if leave_current_mmap == 1:
                    done = True
                    reward = 500#+500
                    # print("+++++++++++++++++++++ successful pass intersection")

                elif leave_current_mmap == 2:
                    done = True
                    # print("+++++++++++++++++++++ restart by code")
                reward = reward / 500
                print("reward=", reward)

                if q_value - rule_q > threshold:
                    print("kick in!！！!！!！!！!！!！!") 
                
                # self.record_rl_intxt(action, q_value, RLpointx, RLpointy, rule_q, collision, leave_current_mmap, ego_s, threshold)
                return np.array(self.state), reward, done,  {}, np.array(self.rule_based_action)

            except:
                print("RL cannot receive an state")

                continue
            
    def record_rl_intxt(self, action, q_value, RLpointx, RLpointy, rule_q, collision, leave_current_mmap, ego_s, threshold):
        fw = open("/home/carla/openai_baselines_update/zwt_ddpg/test_data/record_rl.txt", 'a')   
        fw.write(str(action[0]))   
        fw.write(",")   
        fw.write(str(action[1]))
        fw.write(",")   
        fw.write(str(q_value))
        fw.write(",")   
        fw.write(str(RLpointx))
        fw.write(",")   
        fw.write(str(RLpointy))
        fw.write(",")   
        fw.write(str(rule_q))
        fw.write(",")   
        fw.write(str(collision))
        fw.write(",")   
        fw.write(str(leave_current_mmap))
        fw.write(",")   
        fw.write(str(ego_s))   
        fw.write(",")   

        if q_value - rule_q > threshold:
            fw.write("kick in")  
            print("kick in!！！!！!！!！!！!！!") 
        fw.write("\n")
        fw.close()

    def reset(self, **kargs):
       
        # receive state
        # if the received information meets requirements
        while True:
            try:
                action = [2333,2333,0,0]
                print("-------------",type(action),action)

                self.sock_conn.sendall(msgpack.packb(action))
                print("-------------try received msg in reset")

                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in reset",received_msg)

                self.state = received_msg[0:16]
                collision = received_msg[16]
                leave_current_mmap = received_msg[17]
                RLpointx = received_msg[18]
                RLpointy = received_msg[19]
                self.rule_based_action = [(RLpointx,RLpointy)]

                return np.array(self.state), np.array(self.rule_based_action)

                # if not collision and not leave_current_mmap:
            except:
                print("------------- not received msg in reset")
                collision = 0
                leave_current_mmap = 0
                RLpointx = 0
                RLpointy = 0 - 15/3.6
                self.rule_based_action = [(RLpointx,RLpointy)]

                return np.array(self.state), np.array(self.rule_based_action)

        return np.array(self.state), np.array(self.rule_based_action)


    def render(self, mode='human'):
        # if mode == 'human':
        #     screen_width = 600
        #     screen_height = 400
        #     #world_width = self.problem.xrange
        #     super(MyEnv, self).render(mode=mode)
        pass


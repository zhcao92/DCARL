from __future__ import print_function

import socket
import msgpack

import sys
import math
import numpy as np
import networkx as nx
import gym
import matplotlib.pyplot as plt
import argparse
import logging
import random
import time
import collections
import datetime
import glob
import os
import re
import weakref
import matplotlib.pyplot as plt


from gym import error, spaces, utils
from gym.utils import seeding

##########################################

class ZZZCarlaEnv_lane(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, zzz_client="127.0.0.1", port=2345, recv_buffer=4096):

        self.action_space = spaces.Discrete(8)
        """
        action space:
        0: rule-based policy
        1: emergency brake (acc = -10)
        2: acc = 0; target to outside
        3: acc = 0; target to inside
        4: acc = 1; target to outside
        5: acc = 1; target to inside
        6: acc = -1; target to outside
        7: acc = -1; target to inside
        """
        self._restart_motivation = 0
        self.state = None
        self.steps = 1
        self.collision_times = 0
        self.long_time_collision_flag = False
        self.long_time_collision_times = 0
        self.kick_in_times = 0
        # self._restart_motivation = 0
        # self.state = []
        # self.steps = 1
        # self.collision_times = 0
        self.state_dimention = 20
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((zzz_client, port))
        self.sock.listen()
        self.sock_conn = None
        self.sock_buffer = recv_buffer
 
        low  = np.array([-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50])
        high = np.array([50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()

        self.sock_conn, addr = self.sock.accept()
        print("ZZZ connected at {}".format(addr))

    def step(self, action):
        action = action.astype(int)
        action = action.tolist()
        # send action to zzz planning module
        print("-------------",type(action),action)
        while True:
            try:
                self.sock_conn.sendall(msgpack.packb(action))
                # wait next state
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in step")
                self.state = received_msg[0:20]
                collision = received_msg[20]

                # calculate reward
                reward = 0
            
                # judge if finish
                done = False

                collision_happen = False
                if collision:
                    self.long_time_collision_flag = True
                    self.collision_times += 1
                    print("total_collision:",self.collision_times)
                    collision_happen = True
                    reward = -1
                    done = True

                steps = self.steps
                self.steps = steps + 1
                
                return np.array(self.state), reward, done, collision_happen

            except:
                print("RL cannot receive an state")
                continue

        return np.array(self.state), reward, done, collision_happen


    def reset(self, **kargs):
        # receive state
        # if the received information meets requirements
        while True:
            try:
                action = [0]
                print("-------------",type(action),action)

                self.sock_conn.sendall(msgpack.packb(action))
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                print("-------------received msg in reset",received_msg)

                self.state = received_msg[0:20]
                collision = received_msg[20]
                return np.array(self.state)

            except ValueError:
                continue

        self.steps = 1   
        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            #world_width = self.problem.xrange
            super(MyEnv, self).render(mode=mode)

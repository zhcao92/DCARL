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

class ZZZCarlaEnv(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, zzz_client="127.0.0.1", port=2345, recv_buffer=4096):

        self.action_space = spaces.Discrete(10)
        self._restart_motivation = 0
        self.state = []
        self.steps = 1
        self.collision_times = 0
        self.state_dimention = 14
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((zzz_client, port))
        self.sock.listen()
        self.sock_conn = None
        self.sock_buffer = recv_buffer
 
        low  = np.array([0,  0,   0, 0,  0, -100, 0,  0,   0, 0,  0, -100, 0,  0])
        high = np.array([1, 17, 100, 1, 12,    0, 1, 12, 100, 1, 12,    0, 1, 12])
        
        # ego state: ego_y(0), ego_v(1) 
        # lane 0 : fv_x(2), fv_y(3), fv_v(4), rv_x(5), rv_y(6), rv_v(7)
        # lane 1 : fv_x(8), fv_y(9), fv_v(10), rv_x(11), rv_y(12), rv_v(13)
        # lane 2:...
        # State space = 2+6*lane_num

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()

        self.sock_conn, addr = self.sock.accept()
        print("ZZZ connected at {}".format(addr))

    def step(self, action):

        # send action to zzz planning module
        self.sock_conn.sendall(msgpack.packb(int(action)))
        
        # wait next state
        received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
        self.state = received_msg[0:14]
        collision = received_msg[14]
        leave_current_mmap = received_msg[15]
    
        # calculate reward
        reward = 1

        if collision:
            reward = 0

        # judge if finish
        done = False

        if collision:
            done = True
        
        if leave_current_mmap:
            done = True

        return np.array(self.state), reward, done, {}


    def reset(self, **kargs):
       
        # receive state
        # if the received information meets requirements
        while True:
            try:
                received_msg = msgpack.unpackb(self.sock_conn.recv(self.sock_buffer))
                self.state = received_msg[0:14]
                collision = received_msg[14]
                leave_current_mmap = received_msg[15]
                if not collision and not leave_current_mmap:
                    break
            except ValueError:
                continue
            
        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            #world_width = self.problem.xrange
            super(MyEnv, self).render(mode=mode)

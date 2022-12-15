import os
import math

import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.interpolate import *

import bisect
import unittest

import rospy
import utm
import rosbag
import argparse

import tf

import queue


data = np.loadtxt('road9_0.txt',delimiter=',')

ref_data = np.loadtxt('road9_1.txt',delimiter=',')

flg, ax = plt.subplots(1)
plt.plot(data[:,0], data[:,1], "blue", label="input")
plt.plot(ref_data[:,0], ref_data[:,1], "red", label="ref")
# plt.plot(x_array, y_array, color="blue", label="gps-track")
# plt.plot()

# plt.grid(True)
plt.axis("equal")
# plt.xlabel("x[m]")
# plt.ylabel("y[m]")
# plt.legend()
plt.show()
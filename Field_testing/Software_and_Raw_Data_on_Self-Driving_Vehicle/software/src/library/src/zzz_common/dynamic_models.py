'''
This module gives implementations of motion models and observation models.
'''

import numpy as np
from zzz_common.geometry import wrap_angle
from scipy.special import fresnel

########## Motion models for tracking where inputs are unknown ##########

def motion_BR(state, dt):
    '''
    Brownian Motion model.
    Mean of brownian motion will not change while uncertainty goes larger

    :param state: original state
    :param float dt: time difference after last update (not actually used here)
    :return: updated state
    :rtype: The same type as input state
    '''

    return np.copy(state)

def motion_CV(state, dt):
    '''
    Constant Velocity model

    :param state: original state in format [x, y, vx, vy]
    :type state: np.ndarray or list
    :param float dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    state = np.copy(state)
    state[0] += state[2] * dt
    state[1] += state[3] * dt
    return state

def motion_CA(state, dt):
    raise NotImplementedError()

def motion_CTRV(state, dt):
    raise NotImplementedError()

def motion_CTRA(state, dt):
    '''
    Constant Turn-Rate and (longitudinal) Acceleration model.
    This model also assume that velocity is the same with heading angle.
    CV, CTRV can be modeled by assume value equals zero

    :param state: original state in format [x, y, theta, v, a, w]
    :type state: np.ndarray or list, contents are [x, y, vx, vy]
    :param dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    x, y, th, v, a, w = state
    nth = wrap_angle(th + w * dt)
    nv = v + a * dt
    if np.isclose(w, 0):
        nx = x + (nv + v)/2 * np.cos(th) * dt
        ny = y + (nv + v)/2 * np.sin(th) * dt
    else:
        nx = x + ( nv*w*np.sin(nth) + a*np.cos(nth) - v*w*np.sin(th) - a*np.cos(th)) / (w*w)
        ny = y + (-nv*w*np.cos(nth) + a*np.sin(nth) + v*w*np.cos(th) - a*np.sin(th)) / (w*w)

    state = np.copy(state)
    state[:4] = (nx, ny, nth, nv)
    return state

def motion_CSAA(state, dt):
    '''
    Constant Steering Angle and Acceleration model.

    :param state: original state in format [x, y, theta, v, a, c]
                                           [0  1    2    3  4  5]
    :type state: np.ndarray or list, contents are [x, y, vx, vy]
    :param dt: time difference after last update
    :return: updated state
    :rtype: np.ndarray
    '''

    x, y, th, v, a, c = state
    
    gamma1 = (c*v*v)/(4*a) + th
    gamma2 = c*dt*v + c*dt*dt*a - th
    eta = np.sqrt(2*np.pi)*v*c
    zeta1 = (2*a*dt + v)*np.sqrt(c/2*a*np.pi)
    zeta2 = v*np.sqrt(c/2*a*np.pi)
    sz1, cz1 = fresnel(zeta1)
    sz2, cz2 = fresnel(zeta2)
    
    nx = x + (eta * (np.cos(gamma1)*cz1 + np.sin(gamma1)*sz1 - np.cos(gamma1)*cz2 - np.sin(gamma1)*sz2) +
        2*np.sin(gamma2)*np.sqrt(a*c) + 2*np.sin(th)*np.sqrt(a*c)) / 4*np.sqrt(a*c)*c
    ny = y + (eta * (-np.cos(gamma1)*sz1 + np.sin(gamma1)*cz1 - np.sin(gamma1)*cz2 - np.cos(gamma1)*sz2) +
        2*np.cos(gamma2)*np.sqrt(a*c) - 2*np.sin(th)*np.sqrt(a*c)) / 4*np.sqrt(a*c)*c
    nth = wrap_angle(th - c*dt*dt*a/2 - c*dt*v)
    nv = v + a*dt

    state = np.copy(state)
    state[:4] = (nx, ny, nth, nv)
    return state

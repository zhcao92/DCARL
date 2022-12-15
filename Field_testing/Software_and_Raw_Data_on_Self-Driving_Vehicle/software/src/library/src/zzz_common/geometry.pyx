#cython: embedsignature=True, annotation_typing=False

cimport cython
from libc.math cimport sqrt, ceil

import numpy as np
cimport numpy as np
import numpy.linalg as npl

from shapely.geometry import Polygon

cpdef wrap_angle(float theta):
    '''
    Normalize the angle to [-pi, pi]

    :param float theta: angle to be wrapped
    :return: wrapped angle
    :rtype: float
    '''

    return (theta + np.pi) % (2*np.pi) - np.pi

cpdef dist_from_point_to_line2d(float x0, float y0, float x1, float y1, float x2, float y2):
    '''
    Calculate `distance from point to a oriented line <https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points>` (2D)

    :param (x0,y0): target point
    :type (x0,y0): float, float
    :param (x1,y1): start point of the line
    :type (x1,y1): float, float
    :param (x2,y2): end point of the line
    :type (x2,y2): float, float
    :return: (distance_to_line, point1_to_foot_point, foot_point_to_point2)
    :rtype: (float, float, float), the values are signed (i.e. minus if obtuse angle)

    Note: the distance is negative if the point is at left hand side of the direction of line (p1 -> p2)
    '''

    cdef float l, dl, d1, d2
    l = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
    
    # if two points are the same one
    if l == 0:
       dl = sqrt((y0-y1)*(y0-y1) + (x0-x1)*(x0-x1))
       return dl, 0, 0

    dl = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - x1*y2) / l
    d1 = (x1*x1+x0*(x2-x1)-x1*x2 + y1*y1+y0*(y2-y1)-y1*y2) / l
    d2 = (x2*x2-x0*(x2-x1)-x1*x2 + y2*y2-y0*(y2-y1)-y1*y2) / l
    
    return dl, d1, d2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dist_from_point_to_polyline2d(float x0, float y0, np.ndarray line, bint return_end_distance=False):
    """
    Calculate distance from point to a polyline (2D)

    :param (x0,y0): target point
    :type (x0,y0): float, float
    :param line: target polyline
    :type line: np.ndarray, a Nx2 point array
    :return: (distance_to_line, closest_point_index, closest_point_type); if return_end_distance, then also return (line_head_to_foot_point, foot_point_to_line_end)
    :rtype: float, all returned distances are signed
    """

    if len(line) < 2:
        raise ValueError("Cannot calculate distance to an empty line or a single point!")

    cdef:
        np.ndarray dist_line = npl.norm(line - [x0, y0], axis=1) # dist from current point (x0, y0) to line points
        int closest_idx = np.argmin(dist_line) # index of closet point

        int length = len(line)
        int closest_type = 0 # 0: closest point, 1: next line of closest point, -1: previous line of closest point
        float dist_closest # The actual closest distance from point to polyline
        float dist_previous, dist_previous_head, dist_previous_tail # distance from point to previous line, previous line head, previous line tail
        float dist_next, dist_next_head, dist_next_tail # distance from point to next line, next line head, next line tail
        float dist_start, dist_end # distance to polyline start and polyline end

    # Calculate distances mentioned above
    if closest_idx == 0: # When this point is at line start
        dist_next, dist_next_head, dist_next_tail = dist_from_point_to_line2d(x0, y0,
            line[0,0], line[0,1], line[1,0], line[1,1])
        if dist_next_head < 0: # case 1
            dist_closest = dist_line[closest_idx] # out of the start of the lane to get an obtuse angle
            #jxy: add sign here
            if dist_next < 0:
                dist_closest *= -1
        else: # case 2
            dist_closest = dist_next
            closest_type = 1
    elif closest_idx == length - 1: # When this point is at end
        dist_previous, dist_previous_head, dist_previous_tail = dist_from_point_to_line2d(x0, y0,
            line[length-2, 0], line[length-2, 1], line[length-1, 0], line[length-1, 1])
        if dist_previous_tail < 0: # case 3
            dist_closest = dist_line[closest_idx] # out of the end of the lane to get an obtuse angle
            #jxy: add sign here
            if dist_previous < 0:
                dist_closest *= -1
        else: # case 4
            dist_closest = dist_previous
            closest_type = -1
    else:
        dist_previous, dist_previous_head, dist_previous_tail = dist_from_point_to_line2d(x0, y0,
            line[closest_idx-1, 0], line[closest_idx-1, 1],
            line[closest_idx, 0], line[closest_idx, 1])
        dist_next, dist_next_head, dist_next_tail = dist_from_point_to_line2d(x0, y0,
            line[closest_idx, 0], line[closest_idx, 1],
            line[closest_idx+1, 0], line[closest_idx+1, 1])
        if dist_previous_tail < 0 and dist_next_head < 0: # case 5
            dist_closest = dist_line[closest_idx]
            # sign determination
            if dist_from_point_to_line2d(
                line[closest_idx+1, 0], line[closest_idx+1, 1],
                line[closest_idx-1, 0], line[closest_idx-1, 1],
                line[closest_idx, 0], line[closest_idx, 1])[0] > 0:
                dist_closest *= -1
        elif dist_previous_tail < 0: # case 6
            dist_closest = dist_next
            closest_type = 1
        elif dist_next_head < 0: # case 7
            dist_closest = dist_previous
            closest_type = -1
        else: # case 8
            if abs(dist_next) > abs(dist_previous):
                dist_closest = dist_previous
                closest_type = -1
            else:
                dist_closest = dist_next
                closest_type = 1

    # Return early if total distance is not needed
    if not return_end_distance:
        return (dist_closest, closest_idx, closest_type)

    # Calculate accumulated distance
    if closest_type == 1:
        # closer to next line segment
        dist_start = dist_next_head + np.sum(npl.norm(np.diff(line[:closest_idx+1], axis=0), axis=1))
        dist_end = dist_next_tail + np.sum(npl.norm(np.diff(line[closest_idx+1:], axis=0), axis=1))
    elif closest_type == -1:
        # closer to previous line segment
        dist_start = dist_previous_head + np.sum(npl.norm(np.diff(line[:closest_idx], axis=0), axis=1))
        dist_end = dist_previous_tail + np.sum(npl.norm(np.diff(line[closest_idx:], axis=0), axis=1))
    else:
        # closer to current point
        dist_start = np.sum(npl.norm(np.diff(line[:closest_idx+1], axis=0), axis=1))
        dist_end = np.sum(npl.norm(np.diff(line[closest_idx:], axis=0), axis=1))

    return dist_closest, closest_idx, closest_type, dist_start, dist_end

def dense_polyline2d(np.ndarray line, float resolution, str interp="linear"):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """
    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    if interp != "linear":
        raise NotImplementedError("Other interpolation method is not implemented!")

    cdef np.ndarray s
    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0],s])
    cdef int num = <int>ceil(s[-1]/resolution)

    cdef np.ndarray s_space = np.linspace(0,s[-1],num = num)
    cdef np.ndarray x = np.interp(s_space,s,line[:,0])
    cdef np.ndarray y = np.interp(s_space,s,line[:,1])

    return np.array([x,y]).T

def dense_polyline2d_withvelocity(np.ndarray line, np.ndarray velocity, float resolution, str interp="linear"):
    """
    Dense a polyline by linear interpolation.

    :param resolution: the gap between each point should be lower than this resolution
    :param interp: the interpolation method
    :return: the densed polyline
    """
    if line is None or len(line) == 0:
        raise ValueError("Line input is null")

    if interp != "linear":
        raise NotImplementedError("Other interpolation method is not implemented!")

    cdef np.ndarray s
    s = np.cumsum(npl.norm(np.diff(line, axis=0), axis=1))
    s = np.concatenate([[0],s])
    cdef int num = <int>ceil(s[-1]/resolution)

    cdef np.ndarray s_space = np.linspace(0,s[-1],num = num)
    cdef np.ndarray x = np.interp(s_space,s,line[:,0])
    cdef np.ndarray y = np.interp(s_space,s,line[:,1])
    cdef np.ndarray waypoints_velocity = np.interp(s_space,s,velocity)

    return np.array([x,y]).T, waypoints_velocity

def polygon_iou(p1, p2):
    """
    Intersection area / Union area of two polygons
    """

    p1, p2 = Polygon(p1), Polygon(p2)
    pi = p1.intersection(p2).area
    pu = p1.area + p2.area - pi
    return pi / pu

def box_to_corners_2d(xy, wh, yaw):
    """
    Convert a oriented box to corners on 2d plane.
    Returned box is in counterclock order
    """

    rot_yaw = np.array([
        [ np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])
    x, y = xy
    dx, dy = np.dot(rot_yaw, np.reshape(wh, (-1,1)) / 2)
    return [(x+dx, y+dy), (x-dx, y+dy), (x-dx, y-dy), (x+dx, y-dy)]

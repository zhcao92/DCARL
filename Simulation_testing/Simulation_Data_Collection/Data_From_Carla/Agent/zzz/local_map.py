

import os, sys
import io
import subprocess
from collections import deque
import tempfile
import math
import time
import carla #read lane marker directly from carla

import numpy as np
from gym_carla.envs.tools import dense_polyline2d, dist_from_point_to_polyline2d
from gym_carla.envs.msgs.Lane import Lane
from gym_carla.envs.msgs.LanePoint import LanePoint
from gym_carla.envs.msgs.LaneBoundary import LaneBoundary
from gym_carla.envs.msgs.Map import Map
from gym_carla.envs.msgs.LaneSituation import LaneSituation
from geometry_msgs.msg import PoseStamped, Point32
from nav_msgs.msg import Path

from threading import Lock

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("Please declare environment variable 'SUMO_HOME' (e.g. /usr/share/sumo)")
import sumolib

global wait_to_update_map_count
wait_to_update_map_count = 0

class LocalMap(object):
    def __init__(self, offset_x=0, offset_y=0):
        self._ego_vehicle_x = None # location in meters
        self._ego_vehicle_y = None

        self._ego_vehicle_x_buffer = None
        self._ego_vehicle_y_buffer = None

        self._hdmap = None # HDMap instance loaded from SUMO

        self._offset_x = offset_x # map center offset in meters
        self._offset_y = offset_y

        self._current_edge_id = None # road id string
        self._reference_lane_list = deque(maxlen=20000)
        self._reference_path = []
        self._lane_search_radius = 4

        self._in_section_flag = None
        self._near_section_flag = None

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self._world = client.get_world()

    def setup_hdmap(self, file=None, content=None, mtype=None):
        
        with open("/home/icv/bisim_representation/bisim_representation/gym_carla/envs/map.xodr", 'w') as fw:  
            fw.write(self._world.get_map().to_opendrive())   
        
        converted_file = os.path.join(tempfile.gettempdir(), "sumomap.temp")
        command = ['netconvert', "--opendrive-files", "gym_carla/envs/map.xodr",
                "--opendrive.import-all-lanes", "false",
                "--offset.disable-normalization", "true",
                "--opendrive.curve-resolution", "0.5",
                "-o", converted_file]
        command = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        exitcode = command.wait()
        if exitcode != 0:
            print("SUMO netconvert failed, (exit code: %d)" % exitcode)
        self._hdmap = sumolib.net.readNet(converted_file)
        self._offset_x, self._offset_y = self._hdmap.getLocationOffset()
        print("SUMO: Finish convert")
        
        # Clean temporary files
        os.remove(converted_file)
        return True

    def setup_reference_lane_list(self, reference_path):
        '''
        Generate a list of ids of all lane of the reference path
        jxy: add a basic function of reading reference path
        '''
        if self._hdmap is None:
            return False

        assert type(reference_path)==Path
        self._reference_lane_list.clear()
        self._reference_path = []

        self._reference_path = [(waypoint.pose.position.x, waypoint.pose.position.y) 
                                        for waypoint in reversed(reference_path.poses)]
        self._reference_path.reverse()

        # todo this loop needs to be optimised later

        '''
        for wp in reference_path.poses:
            # TODO: Takes too much time for processing
            print("running-------------------")
            
            wp_map_x, wp_map_y = self.convert_to_map_XY(wp.pose.position.x, wp.pose.position.y)

            # Find the closest lane of the reference path points
            lanes = self._hdmap.getNeighboringLanes(wp_map_x, wp_map_y, self._lane_search_radius, includeJunctions=False)
            if len(lanes) > 0:
                _, closestLane = min((dist, lane) for lane, dist in lanes)
                # Discard duplicate lane ids
                if len(self._reference_lane_list) != 0 and closestLane.getID() == self._reference_lane_list[-1].getID():
                    continue
                self._reference_lane_list.append(closestLane)
        '''

        return True

    def convert_to_map_XY(self, x, y):
        map_x = x # + self._offset_x
        map_y = y # + self._offset_y
        return map_x, map_y

    def convert_to_origin_XY(self, map_x, map_y):
        x = map_x # - self._offset_x
        y = map_y # - self._offset_y
        return x,y

    def receive_new_pose(self, x, y):
        self._ego_vehicle_x_buffer = x
        self._ego_vehicle_y_buffer = y

    def update(self):
        self._ego_vehicle_x = self._ego_vehicle_x_buffer
        self._ego_vehicle_y = self._ego_vehicle_y_buffer
        update_mode = self.should_update_static_map()
        print("[DEBUG] : Update mode",update_mode)
        if update_mode != 0:
            self.update_static_map(update_mode)
            return self.static_local_map
        return None

    def should_update_static_map(self, perception_range_demand = 10, lane_end_dist_thres = 5):
        '''
        Determine whether map updating is needed.
        '''
        map_x, map_y = self.convert_to_map_XY(self._ego_vehicle_x, self._ego_vehicle_y)
        lanes = self._hdmap.getNeighboringLanes(map_x, map_y, self._lane_search_radius, includeJunctions=False)
        if len(lanes) > 0:
            _, closestLane = min((dist, lane) for lane, dist in lanes)
            new_edge = closestLane.getEdge()
            new_edge.id = new_edge.getID()
            # print("Found ego vehicle neighbor edge id = ",new_edge.id)
            if self._current_edge_id is None or new_edge.id != self._current_edge_id:
                # print("Should update static map, edge id ", self._current_edge_id, new_edge.id)
                self._in_section_flag = 0
                self._near_section_flag = 0
                return 1
            else:
                # print("We are in the road, edge id is not changed, edge id is ", self._current_edge_id)
                # if near the section, load the shape of the section
                lane_tail_point = closestLane.getShape()[-1]
                dist_to_lane_tail = math.sqrt(math.pow((map_x - lane_tail_point[0]), 2) + math.pow((map_y - lane_tail_point[1]), 2))
                if dist_to_lane_tail < perception_range_demand:
                    # print("We are near the junction, load the junction")
                    if dist_to_lane_tail < lane_end_dist_thres:
                        # cognition demand, this should be regarded as vehicle in junction
                        if self._in_section_flag == 0:
                            self._in_section_flag = 0.5
                            return 4
                    else:
                        if self._near_section_flag == 0:
                            self._near_section_flag = 1
                            return 3
                else:
                    # in lanes, far from the next junction
                    self._in_section_flag = 0

        if len(lanes) == 0:
            self._near_section_flag = 0
            if self._in_section_flag != 0 and self._in_section_flag != 1:
                #just enter the section
                self._in_section_flag = 1
                return 2
        return 0

    def init_static_map(self):
        '''
        Generate null static map
        '''
        init_static_map = Map()
        init_static_map.in_junction = True
        init_static_map.target_lane_index = -1
        return init_static_map

    def update_static_map(self, update_mode):
        ''' 
        Update information in the static map if current location changed dramatically
        '''
        print("[Dynamic_map]: Updating static map")
        self.static_local_map = self.init_static_map() ## Return this one
        
        # jxy: optimized repeated calculation
        if update_mode == 1:
            self.update_lane_list()
            self.update_target_lane()
        if update_mode == 3:
            self.update_lane_list()
            self.update_target_lane()
            self.update_next_junction()
        if update_mode == 2:
            self.update_junction()
            self.update_next_lanes()
        if update_mode == 4:
            self.update_junction()

        if not self.static_local_map.in_junction:
            self.calibrate_lane_index() # make the righest lane index 0

        # print("Updated static map info: lane_number = %d, in_junction = %d, current_edge_id = %s, target_lane_index = %s",
        #     len(self.static_local_map.lanes), int(self.static_local_map.in_junction),
        #     self._current_edge_id, self.static_local_map.target_lane_index)

    def update_next_lanes(self, step_length = 20):

        map_x, map_y = self.convert_to_map_XY(self._ego_vehicle_x, self._ego_vehicle_y)
        # print("Enter junction, see the next road section")
        if len(self._reference_path) > 1:
            _, nearest_idx, _ = dist_from_point_to_polyline2d(
                map_x, map_y, np.array(self._reference_path)
            )
            last_edge = None
            for i in range(nearest_idx, len(self._reference_path), step_length):
                point = self._reference_path[i]
                lanes = self._hdmap.getNeighboringLanes(point[0], point[1], self._lane_search_radius, includeJunctions=False)
                if len(lanes) > 0:
                    _, closestLane = min((dist, lane) for lane, dist in lanes)
                    current_edge = closestLane.getEdge()
                    if i == nearest_idx:
                        # the nearest point is in the edge, so the vehicle is still in the last edge
                        last_edge = current_edge.getID
                    elif last_edge is None or last_edge != current_edge.getID:

                        lanes_in_edge = current_edge.getLanes()
                        for lane in lanes_in_edge:
                            connections_outgoing = lane.getOutgoing()
                            # Remove fake lane, TODO(zyxin): Remove using SUMO properties (green verge lanes, http://sumo.sourceforge.net/pydoc/sumolib.net.lane.html)
                            if len(connections_outgoing) < 1:
                                continue
                            lane_wrapped = self.wrap_lane(lane)
                            self.static_local_map.next_lanes.append(lane_wrapped)

                        return

    def update_junction(self, closest_dist=100):
        
        self.static_local_map.in_junction = True
        map_x, map_y = self.convert_to_map_XY(self._ego_vehicle_x, self._ego_vehicle_y)
        self.static_local_map.drivable_area.points = []
        nodes = self._hdmap.getNodes() #sumo style
        node_id = 0
        for i in range(len(nodes)):
            e = nodes[i]
            d = sumolib.geomhelper.distancePointToPolygon((map_x, map_y), e.getShape())
            if d < closest_dist:
                node_id = i
                closest_dist = d

        for node_point in nodes[node_id].getShape():
            point = Point32()
            point.x = node_point[0]
            point.y = node_point[1]

            self.static_local_map.drivable_area.points.append(point)

    def update_next_junction(self, closest_dist=100):
        
        self.static_local_map.in_junction = False
        map_x, map_y = self.convert_to_map_XY(self._ego_vehicle_x, self._ego_vehicle_y)
        lanes = self._hdmap.getNeighboringLanes(map_x, map_y, self._lane_search_radius, includeJunctions=False)
        _, closestLane = min((dist, lane) for lane, dist in lanes)
        lane_tail_point = closestLane.getShape()[-1]
        
        self.static_local_map.drivable_area.points = []
        nodes = self._hdmap.getNodes() #sumo style
        node_id = 0
        for i in range(len(nodes)):
            e = nodes[i]
            d = sumolib.geomhelper.distancePointToPolygon((lane_tail_point[0], lane_tail_point[1]), e.getShape())
            if d < closest_dist:
                node_id = i
                closest_dist = d

        for node_point in nodes[node_id].getShape():
            point = Point32()
            point.x = node_point[0]
            point.y = node_point[1]

            self.static_local_map.next_drivable_area.points.append(point)        

    def update_lane_list(self):
        '''
        Update lanes when a new road is encountered
        '''

        
        map_x, map_y = self.convert_to_map_XY(self._ego_vehicle_x, self._ego_vehicle_y)
        lanes = self._hdmap.getNeighboringLanes(map_x, map_y, self._lane_search_radius, includeJunctions=False)
        if len(lanes) > 0:
            self.static_local_map.in_junction = False

            _, closestLane = min((dist, lane) for lane, dist in lanes)
            self.new_lane = closestLane
            self.new_edge = closestLane.getEdge()

            self._current_edge_id = self.new_edge.getID()
            lanes_in_edge = self.new_edge.getLanes()
            for lane in lanes_in_edge:
                connections_outgoing = lane.getOutgoing()
                # Remove fake lane, TODO(zyxin): Remove using SUMO properties (green verge lanes, http://sumo.sourceforge.net/pydoc/sumolib.net.lane.html)
                if len(connections_outgoing) < 1:
                    continue
                lane_wrapped = self.wrap_lane(lane)
                self.static_local_map.lanes.append(lane_wrapped)

    def wrap_lane(self, lane):
        '''
        Wrap lane information into ROS message
        '''
        # jxy: deal with each point in the lane center line.
        # Now I will put all the lane boundary line information in.
        # Directly get information from carla with the center line point as the position.

        lane_wrapped = Lane()
        lane_wrapped.index = lane.getIndex()
        # lane_wrapped.width = lane.getWidth()
        last_x = last_y = last_s = last_tangent = None
        count = 0
        for wp in lane.getShape():
            point = LanePoint()
            x, y = self.convert_to_origin_XY(wp[0], wp[1])
            # Calculate mileage
            if last_s is None:
                point.s = 0
                point.tangent = 3.1415927/2 - math.atan2((lane.getShape()[1][1]-y), (lane.getShape()[1][0]-x))
                # the tangent of the first point can be calculated with the second point
            else:
                point.s = last_s + math.sqrt((x-last_x)*(x-last_x) + (y-last_y)*(y-last_y))
                point.tangent = 3.1415927/2 - math.atan2((y-last_y), (x-last_x))
            point.position.x = x
            point.position.y = y

            flag_straight = 1
            if last_tangent is not None and abs(point.tangent - last_tangent) > 0.0873:
                # 5 degree
                flag_straight = 0
            last_tangent = point.tangent
            # TODO: add more lane point info: jxy dealing with this

            # Update
            last_s = point.s
            last_x = point.position.x
            last_y = point.position.y
            lane_wrapped.central_path_points.append(point)

            # jxy: add lane boundary information
            left_bound = LaneBoundary()
            right_bound = LaneBoundary()

            count = count + 1

            if (flag_straight == 0 and count % 3 == 1) or count % 10 == 1 or count == len(lane.getShape()):
                # too slow... cost about 7s, so divide by 10, but distance becomes 10 times (5m). If direction changes, reduce the gap.
                # we start to know the direction at point 2. TODO: actually the direction at point 1 can be gained by considering the next point.

                location = carla.Location()
                location.x = x
                location.y = y
                waypoint = self._world.get_map().get_waypoint(location)

                angle = 3.1415927/2 - last_tangent
                left_bound.boundary_point.s = point.s
                left_bound.boundary_point.tangent = point.tangent # simplified, theoretically it should be the lane boundary direction
                left_bound.boundary_point.position.x = x + waypoint.lane_width/2.0 * math.cos(angle + 3.1415927/2) # notice that the lane width is not constant, it can vary with s
                left_bound.boundary_point.position.y = y + waypoint.lane_width/2.0 * math.sin(angle + 3.1415927/2)
                
                if str(waypoint.left_lane_marking.type) == 'Broken' or str(waypoint.left_lane_marking.type) == 'BrokenBroken':
                    left_bound.boundary_type = left_bound.BOUNDARY_DASHED_WHITE
                elif str(waypoint.left_lane_marking.type) == 'Solid' or str(waypoint.left_lane_marking.type) == 'SolidSolid' or str(waypoint.left_lane_marking.type) == 'BrokenSolid':
                    left_bound.boundary_type = left_bound.BOUNDARY_SOLID_WHITE
                elif str(waypoint.left_lane_marking.type) == 'Curb':
                    left_bound.boundary_type = left_bound.BOUNDARY_CURB
                else:
                    left_bound.boundary_type = left_bound.BOUNDARY_UNKNOWN

                right_bound.boundary_point.s = point.s
                right_bound.boundary_point.tangent = point.tangent # simplified, theoretically it should be the lane boundary direction
                right_bound.boundary_point.position.x = x - waypoint.lane_width/2.0 * math.cos(angle + 3.1415927/2) # notice that the lane width is not constant, it can vary with s
                right_bound.boundary_point.position.y = y - waypoint.lane_width/2.0 * math.sin(angle + 3.1415927/2)
            
                if str(waypoint.right_lane_marking.type) == 'Broken' or str(waypoint.right_lane_marking.type) == 'BrokenBroken':
                    right_bound.boundary_type = right_bound.BOUNDARY_DASHED_WHITE
                elif str(waypoint.right_lane_marking.type) == 'Solid' or str(waypoint.right_lane_marking.type) == 'SolidSolid' or str(waypoint.right_lane_marking.type) == 'BrokenSolid':
                    right_bound.boundary_type = right_bound.BOUNDARY_SOLID_WHITE
                elif str(waypoint.right_lane_marking.type) == 'Curb':
                    right_bound.boundary_type = right_bound.BOUNDARY_CURB
                else:
                    right_bound.boundary_type = right_bound.BOUNDARY_UNKNOWN
                
                lane_wrapped.left_boundaries.append(left_bound)
                lane_wrapped.right_boundaries.append(right_bound)


        return lane_wrapped

    def update_target_lane(self):
        # Skip passed lanes
        while self._reference_lane_list and self._reference_lane_list[0].getEdge().getID() != self._current_edge_id:
            print("Delete a passed edge: %s, current edge = %s",
                self._reference_lane_list[0].getEdge().getID(), self._current_edge_id)
            self._reference_lane_list.popleft()

        # Find the lane in the next edge
        while self._reference_lane_list and self._reference_lane_list[0].getEdge().getID() == self._current_edge_id:
            print("Delete lane with current edge id: %s, current edge = %s",
                self._reference_lane_list[0].getEdge().getID(), self._current_edge_id)
            self._reference_lane_list.popleft() # TODO(zyxin): Check if the logic is correct here

        if self._reference_lane_list:
            target_lane_id = self._reference_lane_list[0].getID()
            print("Detected next lane id: %s",target_lane_id)

            lanes_in_edge = self.new_edge.getLanes()
            for lane in lanes_in_edge:
                connections_outgoing = lane.getOutgoing()
                for connection in connections_outgoing:
                    if connection.getToLane().getID() == target_lane_id:
                        self.static_local_map.target_lane_index = lane.getIndex()
                        print("Finded next target lane id = %s", self.static_local_map.target_lane_index)
                        return

        # FIXME(zhcao): reference path match target lane
        print("Cannot find next target lane")

    def calibrate_lane_index(self):
        '''
        Makes the drivable lanes starts from 0
        '''
        # TODO(zhcao):should consider ego vehicle lane
        first_index = self.static_local_map.lanes[0].index
        for lane in self.static_local_map.lanes:
            lane.index = lane.index - first_index

        if self.static_local_map.target_lane_index >= 0:
            self.static_local_map.target_lane_index = self.static_local_map.target_lane_index - first_index

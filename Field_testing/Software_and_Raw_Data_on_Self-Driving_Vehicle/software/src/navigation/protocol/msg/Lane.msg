# This message describe a lane with polyline representation

# ----- Basic properties -----
# the rightest lane is 0, reference lane is -1
# TODO: remove this field
int16 index

# Speed limit (km/h)
float32 speed_limit

# Length of the lane. Provided for easy computation
float32 length
float32 width

# Whether this lane allow two way traffic
# This can be true when the road is too narrow or in the center turning lane
bool bidirectional # = False

# The situation when lane ends. This field could be updated with dynamic info.
uint8 stop_state # = 0
uint8 STOP_STATE_UNKNOWN = 0
uint8 STOP_STATE_THRU = 1 # e.g. drive through at lane connection, green light
uint8 STOP_STATE_YIELD = 2 # e.g. unprotected left/right turn, flashing yellow light
uint8 STOP_STATE_STOP = 3 # e.g. red light, yellow light
uint8 STOP_STATE_STOP_YIELD = 4 # e.g. stop sign, right turn at red light
uint8 STOP_STATE_STOP_YIELD_ALL_WAY = 5 # e.g. flashing red light, all way stop sign

# ----- Central path representation -----
# The central_path_points field is used when central_path_type is waypoint.
# Otherwise, central_path_coeffs should be used
LanePoint[] central_path_points
float32[] central_path_coeffs

uint8 central_path_type # = 0
uint8 CENTRAL_PATH_WAYPOINT = 0 # discretized
uint8 CENTRAL_PATH_LINE = 1
uint8 CENTRAL_PATH_CONIC = 2 # conic section, including parabola and hyperbola
uint8 CENTRAL_PATH_POLYNOMIAL = 3 #
uint8 CENTRAL_PATH_BEZIER = 4

# ----- Boundary representation -----
# The boundary description of current lane.
# Not that the boundary type only describe the behaviour from current lane to neighbour lane or road shoulder
LaneBoundary[] left_boundaries
LaneBoundary[] right_boundaries

# ----- Auxiliary information of the lane -----
# Road situations on this line. This field could be updated with dynamic info.
LaneSituation[] situations

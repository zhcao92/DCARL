# This message describe the boundary type of a part of a lane

# Lane boundary point
LanePoint boundary_point

# Boundary type from between `s` and `s` of the next section start
uint8 boundary_type
uint8 BOUNDARY_UNKNOWN = 0
uint8 BOUNDARY_DASHED_WHITE = 1  # neighbour lane has same direction.
uint8 BOUNDARY_DASHED_YELLOW = 2 # neighbour lane has different direction.
uint8 BOUNDARY_SOLID_WHITE = 3   # neighbour lane has same direction, not allowed to change lane.
uint8 BOUNDARY_SOLID_YELLOW = 4  # neighbour lane has different direction, not allowed to change lane.
uint8 BOUNDARY_SOLID_YELLOW_TURN = 5  # neighbour lane has different direction, not allowed to change lane unless turning.
uint8 BOUNDARY_CURB = 6          # neighbour is road shoulder

# Confidence of the lane boundary classification
float32 confidence

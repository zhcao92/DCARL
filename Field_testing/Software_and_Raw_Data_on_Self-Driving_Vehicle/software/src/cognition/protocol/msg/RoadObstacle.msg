# This message contains all the information that planning needs for a road object


# Unique indentity of the obstacle
uint64 uid

# The confidence of existence, can be used to determine whether this object is valid, or invalid
# A possible way of get the confidence is by using tracking age
float32 confidence

# The best guess of obstacle type
zzz_perception_msgs/ObjectClass cls

# Estimated kinematic properties
zzz_driver_msgs/RigidBodyState state
# Kinematic properties in Frenet Frame
zzz_driver_msgs/FrenetSerretState2D ffstate

# Relationship to lane, for prediction or locating
float32 lane_index # in which lane
float32 lane_anglediff # angle difference
float32 lane_dist_left_t # distance to lane
float32 lane_dist_right_t
float32 lane_dist_s # s position in the road

# A flag to mark whether the object is static
bool static

# XXX: Do we need history trajectories?

# ----- Physical Boundary (Optional) -----
uint8 shape_type
uint8 SHAPE_UNDEFINED = 0
uint8 SHAPE_POLYGON = 1

# Representation of the object if it's represented by polygon
geometry_msgs/Polygon shape

# bbox, refer to BoundingBox in zzz_perception_msgs

zzz_perception_msgs/DimensionWithCovariance dimension
#uint8 dimension

# Null uncertainty of Inf uncertainty means that the shape is not actually generated
float32[] shape_uncertainty

# ----- High level behavior estimation -----

uint8 behavior
uint8 BEHAVIOR_UNKNOWN = 0
uint8 BEHAVIOR_STOPPING = 1 # vehicle is going to stop completely or stay stopped
uint8 BEHAVIOR_FOLLOW = 2 # follow lane or straight line
uint8 BEHAVIOR_MOVING_LEFT = 3 # changing to its left lane or branch left
uint8 BEHAVIOR_MOVING_RIGHT = 4 # changing to its right lane or branch right

# This field is preserved to adapt to yield situation.
# Yield situation includes left turn, right turn, sequencial stop sign, emergency vehicle, etc.
# Yield means if your path conflict with the object's, you should wait for the object
# to go through the conflict point.
uint8 priority
uint8 PRIORITY_UNKNOWN = 0
uint8 PRIORITY_NORMAL = 1
uint8 PRIORITY_CAUTIOUS = 2 # You can move when this obstacle is far away
uint8 PRIORITY_STOP = 3 # Should let this vehicle to go and then you can move

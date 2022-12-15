# This message represents a homogeneous representation of driving space (static and dynamic map)

Header header

# Ego Vehicle Info
zzz_driver_msgs/RigidBodyState ego_state
# zzz_driver_msgs/FrenetSerretState2D ego_ffstate
float32 ego_lane_index

# Static Info
zzz_navigation_msgs/Lane[] lanes

# Dynamic Info
RoadObstacle[] obstacles

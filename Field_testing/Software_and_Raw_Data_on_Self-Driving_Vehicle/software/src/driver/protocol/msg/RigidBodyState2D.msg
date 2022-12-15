# This message describes rigid body state in 2d plane

# 2D states
float32 x
float32 y
float32 yaw
float32[9] pose_covariance

# First order derivatives
float32 vx
float32 vy
float32 omega
float32[9] twist_covariance

# Second order derivatives
float32 ax
float32 ay
float32 epsilon
float32[9] accel_covariance

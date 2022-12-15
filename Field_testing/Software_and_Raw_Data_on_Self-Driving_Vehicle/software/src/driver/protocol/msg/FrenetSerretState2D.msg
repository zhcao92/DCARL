# This message describes a state in 2d Frenet-Serret Frame
# By 2d Frenet-Serret Frame we ignore the movement in z (in Cartesian) / b (in Frenet) direction
# For more information, refer to https://en.wikipedia.org/wiki/Frenet-Serret_formulas

# 2D states
float32 s # Offset in tanget direction
float32 d # Offset in normal direction
          # we omit the offset in binormal direction
float32 psi # Heading angle in s-d plane
float32[9] pose_covariance

# First order derivatives
float32 vs
float32 vd
float32 omega
float32[9] twist_covariance

# Second order derivatives
float32 sa # prevent keyword conflict
float32 ad
float32 epsilon
float32[9] accel_covariance

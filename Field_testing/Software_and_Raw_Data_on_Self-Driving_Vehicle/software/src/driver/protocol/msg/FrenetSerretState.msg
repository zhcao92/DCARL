# This message describes a state in 3d Frenet-Serret Frame
# Currently we don't use 3D frenet state actually
# For more information, refer to https://en.wikipedia.org/wiki/Frenet-Serret_formulas

# 3D states
float32 s # Offset in tanget direction
float32 d # Offset in normal direction
float32 b # Offset in binormal direction
float32 psi_s
float32 psi_d
float32 psi_b
float32[36] pose_covariance

# First order derivatives
float32 vs
float32 vd
float32 vb
float32 omega_s
float32 omega_d
float32 omega_b
float32[36] twist_covariance

# Second order derivatives
float32 sa # prevent keyword conflict
float32 ad
float32 ab
float32 epsilon_s
float32 epsilon_d
float32 epsilon_b
float32[36] accel_covariance

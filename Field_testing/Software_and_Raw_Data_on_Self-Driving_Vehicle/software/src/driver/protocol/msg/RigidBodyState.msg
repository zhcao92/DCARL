# This message contains commonly used state variables of rigid body

# ID of frame fixed to the rigid body
string child_frame_id

# Location and orientatation of the object
geometry_msgs/PoseWithCovariance  pose

# Linear and angular velocity of the object
geometry_msgs/TwistWithCovariance twist

# Linear and angular acceleration of the object
geometry_msgs/AccelWithCovariance accel

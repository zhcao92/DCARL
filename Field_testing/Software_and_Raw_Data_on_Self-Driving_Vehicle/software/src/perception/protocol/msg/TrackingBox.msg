# Defines a box-shaped 3D tracking result.

# Unique id to determine the identification of tracked object.
# This field could be a increasing number from zero or random hash number
uint64 uid

# Defines the (existence) confidence of the object [-1 ~ 1].
# The confidence can be negative to demonstrate that this object is invalid.
# This confidence should not be fed into decision system. It should be used for validation purpose.
float32 confidence

# Describe several classification result for the object
# This field is required to be sorted in descending order of scores
ObjectClass[] classes

# This field contains the behavior identification based on light signal or hand signal
ObjectSignals signal

# Current 3D bounding box.
BoundingBox bbox

# Estimated 3D velocity and accelaration
geometry_msgs/TwistWithCovariance twist
geometry_msgs/AccelWithCovariance accel

# This field is for store auxiliary text or data
string comments


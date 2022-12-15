# Defines a box-shaped 2D detection result.

# ----------------- Data -----------------

# The 2D data that generated these results (i.e. region proposal cropped out of
#   the image). Not required for all use cases, so it may be empty.
sensor_msgs/Image source_img

# ---------------- Properties --------------------

# Describe several classification result for the object
# This field is required to be sorted in descending order of scores
ObjectClass[] classes

# This field indicates visual (or sound?) signal from the object
ObjectSignals signal

# 2D bounding box surrounding the object.
BoundingBox2D bbox

# This field is for store auxiliary text or data
string comments

# Defines a box-shaped 3D detection result.

# ----------------- Data -----------------

# The 3D data that generated these results (i.e. object region cropped out of
#   the point cloud). This information is not required for all detectors, so it may be empty.
sensor_msgs/PointCloud2 source_cloud

# The 2D data in the same area for fusion purpose
sensor_msgs/Image source_img

# The frame id of source_img
string source_frame

# ---------------- Properties --------------------

# Describe several classification result for the object
# This field is required to be sorted in descending order of scores
ObjectClass[] classes

# This field indicates visual (or sound?) signal from the object
ObjectSignals signal

# 3D bounding box surrounding the object.
BoundingBox bbox

# This field is for store auxiliary text or data. (e.g. annotation notes)
string comments

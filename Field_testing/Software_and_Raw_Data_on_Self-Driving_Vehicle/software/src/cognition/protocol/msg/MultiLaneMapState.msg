# This message describes a multi-lane dynamic map model


# distance before next junction
float32 distance_to_junction

# -------- Multi-lanes info --------
# all lane info, lanes must be ordered from right most lane
LaneState[] lanes

# the lane that ego vehicle drives on
float32 ego_lane_index

# target lane index (0 refer to right-most lane in right-side driving situation)
# TODO: Change to list
int8 target_lane_index

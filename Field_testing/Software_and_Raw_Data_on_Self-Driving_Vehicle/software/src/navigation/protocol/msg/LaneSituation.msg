# This message store any uncommon situations on a road
# Situation can be a map specified object or fresh events

# The location of the situation in a lane (distance from lane start)
float32 s

# How long this situation affect along the lane.
# Inf means take effect until lane ends
float32 length

# Type of situation
uint8 situation_type
uint8 SITUATION_NOT_SPECIFIED = 0
uint8 SITUATION_BLOCKED = 1 # Something blocked this lane, vehicle should stop here
uint8 SITUATION_CROSSWALK = 2 # Crosswalk
uint8 SITUATION_REDUCE_SPEED = 3 # Reducing speed is required. This can be due to speed bump, construction, or school bus stop, etc

# ------- Situation information -------
# The new speed limit. Zero means the vehicle is required to stop
float32 reduced_max_speed

# Additional information
string comments

# 0 = track, 1 = cluster
bool clusterortrack
#track
#Unique track ID;0->65535
uint16 obstacle_id
#Obstacle type - NOT USED
int16 obstacle_class
#Longitudinal displacement(x); unit = m
float32 longitude_displ
 #Lateral displacement(y); unit = m
float32 lateral_displ
#Obstacle length - NOT USED
float32 length
#Obstacle width - NOT USED
float32 width
#Obstacle height - NOT USED
float32 height
#Relative longitudinal velocity; unit = m/s
float32 longitude_vel
#Relative lateral velocity; unit = m/s
float32 lateral_vel
#Relative longitudinal acceleration; unit = m/s2 - NOT USED
float32 longitude_acc
#Relative lateral acceleration; unit = m/s2 - NOT USED
float32 lateral_acc
#Obstacle classify quality - NOT USED
int16 obstacle_class_quality
#Radar cross section; unit = dBm2
float32 rcs_value
#The current lifetime of the track; unit = s
float32 track_lifetime
#Cluster
#Radial distance of the cluster; unit = m
float32 range
#Cluster angle; unit = deg；-90->90
int16 azimuth
#Relative velocity of the cluster; unit = m/s
float32 vrel
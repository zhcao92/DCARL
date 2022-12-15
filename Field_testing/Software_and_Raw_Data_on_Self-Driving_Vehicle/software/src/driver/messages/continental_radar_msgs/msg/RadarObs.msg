# 0 = track, 1 = cluster
bool clusterortrack
# obstacle Id
uint8 obstacle_id
# longitude distance to the radar; (+) = forward; unit = m
float32 longitude_dist
# lateral distance to the radar; (+) = left; unit = m
float32 lateral_dist
# longitude velocity to the radar; (+) = forward; unit = m/s
float32 longitude_vel
# lateral velocity to the radar; (+) = left; unit = m/s
float32 lateral_vel
# obstacle Radar Cross-Section; unit = dBsm
float32 rcs
# 0 = moving, 1 = stationary, 2 = oncoming, 3 = stationary candidate
# 4 = unknown, 5 = crossing stationary, 6 = crossing moving, 7 = stopped
uint16 dynprop
# longitude distance standard deviation to the radar; (+) = forward; unit = m
float32 longitude_dist_rms
# lateral distance standard deviation to the radar; (+) = left; unit = m
float32 lateral_dist_rms
# longitude velocity standard deviation to the radar; (+) = forward; unit =
# m/s
float32 longitude_vel_rms
# lateral velocity standard deviation to the radar; (+) = left; unit = m/s
float32 lateral_vel_rms
# obstacle probability of existence
float32 probexist
# The following is only valid for the track object message
# 0 = deleted, 1 = new, 2 = measured, 3 = predicted, 4 = deleted for, 5 = new
# from merge
uint16 meas_state
# longitude acceleration to the radar; (+) = forward; unit = m/s2
float32 longitude_accel 
# lateral acceleration to the radar; (+) = left; unit = m/s2
float32 lateral_accel
# orientation angle to the radar; (+) = counterclockwise; unit = degree
float32 oritation_angle
# longitude acceleration standard deviation to the radar; (+) = forward; unit
# = m/s2
float32 longitude_accel_rms
# lateral acceleration standard deviation to the radar; (+) = left; unit =
# m/s2
float32 lateral_accel_rms
# orientation angle standard deviation to the radar; (+) = counterclockwise;
# unit = degree
float32 oritation_angle_rms
# obstacle length; unit = m
float32 length 
# obstacle width; unit = m
float32 width  
# 0: point; 1: car; 2: truck; 3: pedestrian; 4: motorcycle; 5: bicycle; 6:
# wide; 7: unknown
uint16 obstacle_class
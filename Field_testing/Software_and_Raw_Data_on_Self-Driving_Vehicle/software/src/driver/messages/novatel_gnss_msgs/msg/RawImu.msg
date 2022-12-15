#Raw_Imu
  Header header

#novatel long header
  LongHeader long_header

  uint32 gps_week

# Seconds of week.
  float64 gps_seconds  

# Status of the IMU. The content varies with IMU type.
  uint32 imuStatus  

# change in velocity along z axis.
  int32 z_velocity_change      

# -change in velocity along y axis.
  int32 y_velocity_change_neg  

# change in velocity along x axis.
  int32 x_velocity_change      

# change in angle around z axis.
  int32 z_angle_change         

# -change in angle around y axis.
  int32 y_angle_change_neg     

# change in angle around x axis.
  int32 x_angle_change         

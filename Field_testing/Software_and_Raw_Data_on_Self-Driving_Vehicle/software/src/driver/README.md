# mini-auto-drivers

## drivers for xiaopeng

### camera

add 'usb_cam' from github: https://github.com/ros-drivers/usb_cam.git

topics published are below(just for left camera, similar for right camera):

/left_usb_cam/camera_info
/left_usb_cam/image_raw
/left_usb_cam/image_raw/compressed
/left_usb_cam/image_raw/compressed/parameter_descriptions
/left_usb_cam/image_raw/compressed/parameter_updates
/left_usb_cam/image_raw/compressedDepth
/left_usb_cam/image_raw/compressedDepth/parameter_descriptions
/left_usb_cam/image_raw/compressedDepth/parameter_updates
/left_usb_cam/image_raw/theora
/left_usb_cam/image_raw/theora/parameter_descriptions
/left_usb_cam/image_raw/theora/parameter_updates

### lidar

the lidar code used is robsenslidar16p, it is for one RS32(middle) and two RS16(left and right) type lidar simultaneously. Topics published for each lidar are similar, example of middle lidar is below:

/middle/cloud_node/parameter_descriptions
/middle/cloud_node/parameter_updates
/middle/node_status
/middle/rslidar_node/parameter_descriptions
/middle/rslidar_node/parameter_updates
/middle/rslidar_packets
/middle/rslidar_packets_difop
/middle/rslidar_points
/middle/skippackets_num
/middle/sync_header

topic '/middle/rslidar_points' is to publish pointcloud.

display in rviz, type frameid : rslidar

add 'Panda40p' from github: https://github.com/HesaiTechnology/Pandar40p_ros.git

Hesai Panda40 lidar is not used on xiaopeng vehicle.

```
git clone https://github.com/HesaiTechnology/Pandar40p_ros.git
cd Pandar40p_ros
git submodule update --init --recursive
```

### imu 

use oxford_gps_eth now. And published topics are below:

/gps/fix
/gps/odom
/gps/vel
/imu/data

add 'xsens_driver' from github: https://github.com/ethz-asl/ethzasl_xsens_driver.git

it is not used up to now.

### can 

can topics are below:

/xp/auto_control
/xp/auto_state
/xp/auto_state_ex
/xp/eps_status
/xp/esc_status

/xp/auto_control subscribes control messages.

other four topics are for status feedback.

### continental_radar 

it is for radar of type ARS408.

continental radar topics are below:

/front/RadarMsg

/rear/RadarMsg

display in rviz, type frameid : continental_radar

### srr_radar 

it is for radar of type SRR208.

not tested up to now

### us_radar 

it is for ultrasound radar. 

ultrasound radar topics are below:

/us_radar/Detections
/us_radar/Distances


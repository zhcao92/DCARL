# rosbag play --topics /tf /left_usb_cam/image_raw/compressed /middle/rslidar_points /zzz/navigation/ego_pose /zzz/perception/objects_tracked  --bag "$1"  -s "$2" 
rosbag play --topics /tf /left_usb_cam/image_raw/compressed /middle/rslidar_packets /middle/rslidar_packets_difop /zzz/navigation/ego_pose /zzz/perception/objects_tracked  --bag "$1"  -s "$2" 

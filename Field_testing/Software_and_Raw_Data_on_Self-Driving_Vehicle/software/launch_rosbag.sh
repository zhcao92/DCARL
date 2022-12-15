export RUN_PLATFORM=xiaopeng
source devel/setup.bash
rosbag play rosbag/case$1.bag --topics /left_usb_cam/image_raw/compressed /tf /tf_static /xp/auto_control /xp/auto_state /xp/auto_state_ex /xp/eps_status /xp/esc_status /zzz/cognition/local_dynamic_map/map_with_ref /zzz/navigation/ego_pose
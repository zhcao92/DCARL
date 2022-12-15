cmake_minimum_required(VERSION 2.8.3)
project(radar_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

## Declare ROS messages and services
add_message_files(DIRECTORY msg FILES
  RadarDetection.msg
  RadarDetectionStamped.msg
  RadarDetectionArray.msg
  RadarTrack.msg
  RadarTrackStamped.msg
  RadarTrackArray.msg
  # high level Tx
  RadarStatus.msg
  RadarErrorStatus.msg
)

## Generate added messages and services
generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Declare a catkin package
catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
)


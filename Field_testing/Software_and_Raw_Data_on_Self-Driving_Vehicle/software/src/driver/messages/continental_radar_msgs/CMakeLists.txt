cmake_minimum_required(VERSION 2.8.3)
project(continental_radar_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES
  RadarMsg.msg
  RadarObs.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)


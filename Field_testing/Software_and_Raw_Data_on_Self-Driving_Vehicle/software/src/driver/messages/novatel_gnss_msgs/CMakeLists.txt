cmake_minimum_required(VERSION 2.8.3)
project(novatel_gnss_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES
  LongHeader.msg
  BdsEphemeris.msg
  BestPosb.msg
  BestVelb.msg
  GloEphemeris.msg
  GpChc.msg
  GpsEphemeris.msg
  Headingb.msg
  InsPva.msg
  RangeIn.msg
  Range.msg
  RawImu.msg
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

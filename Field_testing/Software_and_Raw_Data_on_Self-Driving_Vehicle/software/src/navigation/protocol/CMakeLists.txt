cmake_minimum_required(VERSION 2.8.3)
project(zzz_navigation_msgs)

find_package(catkin REQUIRED genmsg std_msgs actionlib_msgs # essential dependency  
    geometry_msgs sensor_msgs
)

catkin_python_setup()

add_message_files(DIRECTORY msg FILES
    Lane.msg
    LaneBoundary.msg
    LanePoint.msg
    LaneSituation.msg
    Map.msg
    MapString.msg
    ReroutingRequest.msg
)

generate_messages(DEPENDENCIES std_msgs actionlib_msgs geometry_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS std_msgs actionlib_msgs geometry_msgs sensor_msgs)

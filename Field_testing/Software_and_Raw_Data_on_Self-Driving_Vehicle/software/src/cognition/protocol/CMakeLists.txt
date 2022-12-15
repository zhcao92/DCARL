cmake_minimum_required(VERSION 2.8.3)
project(zzz_cognition_msgs)

find_package(catkin REQUIRED genmsg std_msgs # essential dependency  
    geometry_msgs zzz_navigation_msgs zzz_perception_msgs zzz_driver_msgs
)

catkin_python_setup()

add_message_files(DIRECTORY msg FILES
    JunctionMapState.msg
    LaneState.msg
    MultiLaneMapState.msg
    MapState.msg
    RoadObstacle.msg
    DrivingSpace.msg
    DynamicBoundary.msg
    DynamicBoundaryPoint.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs zzz_perception_msgs zzz_navigation_msgs zzz_driver_msgs)
catkin_package(CATKIN_DEPENDS std_msgs geometry_msgs zzz_perception_msgs zzz_navigation_msgs zzz_driver_msgs)

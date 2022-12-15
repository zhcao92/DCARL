cmake_minimum_required(VERSION 2.8.3)
project(zzz_planning_msgs)

find_package(catkin REQUIRED genmsg std_msgs # essential dependency
    nav_msgs
)

add_message_files(DIRECTORY msg FILES
    DecisionTrajectory.msg
    PlannedTrajectory.msg
    VehicleState.msg
)

generate_messages(DEPENDENCIES std_msgs nav_msgs)
catkin_package(CATKIN_DEPENDS std_msgs nav_msgs)

cmake_minimum_required(VERSION 2.8.3)
project(zzz_control_msgs)

find_package(catkin REQUIRED genmsg std_msgs # essential dependency
)

add_message_files(DIRECTORY msg FILES
    ControlCommand.msg
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS std_msgs)

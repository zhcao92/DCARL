cmake_minimum_required(VERSION 2.8.3)
project(zzz_monitor_msgs)

find_package(catkin REQUIRED genmsg std_msgs # essential dependency
)

# catkin_python_setup()

add_message_files(DIRECTORY msg FILES
    DiagnosticInfo.msg
    ModuleBeat.msg
    ModuleStatus.msg
)

generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS std_msgs)

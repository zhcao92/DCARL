cmake_minimum_required(VERSION 2.8.3)
project(zzz_driver_msgs)

catkin_python_setup()

find_package(catkin REQUIRED  
    geometry_msgs message_generation geometry_msgs std_msgs 
)

add_message_files(DIRECTORY msg FILES
    AuxiliaryReport.msg
    ChassisReport.msg
    ControlReport.msg
    FrenetSerretState.msg
    FrenetSerretState2D.msg
    ResourceReport.msg
    RigidBodyState.msg
    RigidBodyStateStamped.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(
    CATKIN_DEPENDS 
    message_runtime 
    std_msgs 
    geometry_msgs
)

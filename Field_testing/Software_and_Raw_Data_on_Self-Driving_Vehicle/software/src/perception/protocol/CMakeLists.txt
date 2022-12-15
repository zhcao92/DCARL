cmake_minimum_required(VERSION 2.8.3)
project(zzz_perception_msgs)

find_package(catkin REQUIRED genmsg std_msgs # essential dependency
    geometry_msgs sensor_msgs
)

add_message_files(DIRECTORY msg FILES
    BoundingBox.msg
    BoundingBox2D.msg
    DetectionBox.msg
    DetectionBox2D.msg
    DetectionBox2DArray.msg
    DetectionBoxArray.msg
    Dimension2DWithCovariance.msg
    DimensionWithCovariance.msg
    LaneDetection.msg
    LaneDetectionArray.msg
    ObjectClass.msg
    ObjectSignals.msg
    Pose2DWithCovariance.msg
    TrackingBox.msg
    TrackingBoxArray.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)
catkin_package(CATKIN_DEPENDS std_msgs geometry_msgs sensor_msgs)


#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

#define STDOUT(str) std::cout << str << std::endl

#define APP_NAME "tf4hmi"

class Tf4hmi {
public:
    Tf4hmi() ;
    ~Tf4hmi() {};
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);

private:
    std::string input_odom_topic_;

    ros::NodeHandle node_handle_;
    ros::NodeHandle private_nh_;
    ros::Subscriber sub_odom_;
    Eigen::Matrix3d lidar2imu_rotation_;
    Eigen::Vector3d lidar2imu_translation_;
};


Tf4hmi::Tf4hmi() : private_nh_("~"){
  private_nh_.param<std::string>("input_odom_topic", input_odom_topic_, "");
  std::string     rotation_str, translation_str;
  private_nh_.param<std::string>("lidar2imu_rotation", rotation_str,
      "[0.9982, -0.04619, -0.038339, 0.046762, 0.99881, 0.014155, 0.03764, -0.015922, 0.99916]");  
  private_nh_.param<std::string>("lidar2imu_translation", translation_str,
      "[0.93, 0.05, 0.95]");  

  YAML::Node rot = YAML::Load(rotation_str);
  YAML::Node trans = YAML::Load(translation_str);
  lidar2imu_rotation_ << rot[0].as<double>(), rot[1].as<double>(), rot[2].as<double>(), 
      rot[3].as<double>(), rot[4].as<double>(), rot[5].as<double>(), 
      rot[6].as<double>(), rot[7].as<double>(), rot[8].as<double>();
  lidar2imu_translation_ << trans[0].as<double>(), trans[1].as<double>(), trans[2].as<double>();

  sub_odom_ = node_handle_.subscribe(input_odom_topic_, 10,
      &Tf4hmi::odomCallback, this,
      ros::TransportHints().reliable().tcpNoDelay(true));
}

void Tf4hmi::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  static tf::TransformBroadcaster br;

  tf::Transform transform;
  Eigen::Quaterniond q_imu(
          msg->pose.pose.orientation.w,     
          msg->pose.pose.orientation.x,
          msg->pose.pose.orientation.y,
          msg->pose.pose.orientation.z );
  Eigen::Vector3d pose_imu(
          msg->pose.pose.position.x,
          msg->pose.pose.position.y,
          msg->pose.pose.position.z);
#if 1
  Eigen::Quaterniond q0(lidar2imu_rotation_);
  q_imu = q_imu * q0;
  pose_imu = pose_imu + q_imu.toRotationMatrix() * lidar2imu_translation_;
#endif
  transform.setOrigin( tf::Vector3(pose_imu.x(), 
      pose_imu.y(), pose_imu.z()) );
  tf::Quaternion q (
    q_imu.x(),
    q_imu.y(), 
    q_imu.z(),
    q_imu.w() );
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, msg->header.stamp, "map", "rslidar"));
}

int main (int argc, char** argv) {
  ros::init(argc, argv, "Tf4hmi");

  Tf4hmi loc;

  // handle callbacks until shut down
  ros::spin();
  return 0;
}
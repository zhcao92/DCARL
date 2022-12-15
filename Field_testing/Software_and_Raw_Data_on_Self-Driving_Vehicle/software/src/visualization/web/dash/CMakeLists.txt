cmake_minimum_required(VERSION 2.8.3)
project(zzz_visualization_web_dash)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
  nodes/dashboard
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

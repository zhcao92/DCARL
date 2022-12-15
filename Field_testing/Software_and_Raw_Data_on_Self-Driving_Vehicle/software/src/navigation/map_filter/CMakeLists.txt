cmake_minimum_required(VERSION 2.8.3)
project(zzz_navigation_map_filter)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS nodes/lane_geometry
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

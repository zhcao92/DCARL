cmake_minimum_required(VERSION 2.8.3)
project(zzz_navigation_map_provider_sumo)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS nodes/local_map_generator
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

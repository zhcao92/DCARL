cmake_minimum_required(VERSION 2.8.3)
project(zzz_planning_decision_safeguard)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
  nodes/reachable_set
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

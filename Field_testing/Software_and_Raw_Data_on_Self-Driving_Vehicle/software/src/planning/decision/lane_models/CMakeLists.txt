cmake_minimum_required(VERSION 2.8.3)
project(zzz_planning_decision_lane_models)

find_package(catkin REQUIRED)
catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
  nodes/idm_lane_utility
  nodes/rls_train
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

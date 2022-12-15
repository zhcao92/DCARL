cmake_minimum_required(VERSION 3.0.0)
project(tf4hmi)
# add_compile_options(std=c++11)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -Wfloat-conversion")


# find_package(CUDA)
find_package(Eigen3)
# find_package(Boost)
# find_package(OpenCV REQUIRED)
# find_package(PCL 1.8 REQUIRED)
# find_package(PCL REQUIRED)
# add_definitions(${PCL_DEFINITIONS})
# link_directories(/usr/lib/x86_64-linux-gnu)


find_package(catkin REQUIRED COMPONENTS
	roscpp
	sensor_msgs
	geometry_msgs
	nav_msgs
	tf
)



catkin_package(
	INCLUDE_DIRS
	# include
	CATKIN_DEPENDS
	roscpp
	nav_msgs
	sensor_msgs
	geometry_msgs
	tf

)

# include_directories(${CUDA_INCLUDE_DIRS})

include_directories(
	${catkin_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
)
# target_compile_options(lidar_cnn_seg_detect_trt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
add_executable(tf4hmi
  src/main.cc
)
target_link_libraries(tf4hmi
	${catkin_LIBRARIES}
	yaml-cpp
)
add_dependencies(tf4hmi
  ${catkin_EXPORTED_TARGETS}
)

install(TARGETS
	tf4hmi
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
	PATTERN ".svn" EXCLUDE
)


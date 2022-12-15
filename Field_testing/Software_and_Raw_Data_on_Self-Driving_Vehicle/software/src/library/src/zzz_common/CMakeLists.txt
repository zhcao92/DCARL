
include(UseCython)
include(FindPythonExtensions)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

include_directories(${NumPy_INCLUDE_DIR})
add_cython_target(geometry CXX)
add_library(geometry MODULE ${geometry})
python_extension_module(geometry)

add_cython_target(kinematics CXX)
add_library(kinematics MODULE ${kinematics})
python_extension_module(kinematics)

set_target_properties(geometry kinematics
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

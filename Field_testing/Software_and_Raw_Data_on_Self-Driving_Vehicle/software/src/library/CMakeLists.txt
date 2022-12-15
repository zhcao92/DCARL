cmake_minimum_required(VERSION 3.3)
project(zzz_common)

find_package(catkin REQUIRED)
catkin_package()

# ===== Find skbuild location and related cmake path =====
find_package(PythonInterp REQUIRED)
execute_process(COMMAND "${PYTHON_EXECUTABLE}"
    -c "import skbuild; print(skbuild.__file__)"
    OUTPUT_VARIABLE _skbuild_init
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _skbuild_status
    ERROR_QUIET
)
if(NOT _skbuild_status EQUAL 0)
    message(FATAL_ERROR "Error when finding scikit-build, which is required for building. Please install scikit-build with pip.")
endif()

get_filename_component(_skbuild_dir ${_skbuild_init} DIRECTORY)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${_skbuild_dir}/resources/cmake)

# ===== Register cython modules =====
add_subdirectory(src/zzz_common)

catkin_python_setup()

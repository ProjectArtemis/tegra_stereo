cmake_minimum_required(VERSION 2.8.3)
project(tegra_stereo)

find_package(catkin REQUIRED
  COMPONENTS roscpp nodelet
  cv_bridge image_geometry image_proc image_transport
  message_filters sensor_msgs stereo_msgs
  camera_calibration_parsers camera_info_manager)
find_package(Boost REQUIRED COMPONENTS thread)

find_package(OpenCV REQUIRED)

find_package(PkgConfig)
pkg_check_modules(YAML_CPP yaml-cpp)
if(${YAML_CPP_VERSION} VERSION_GREATER 0.5)
  add_definitions(-DHAVE_NEW_YAMLCPP)
endif()

# C++11 is required
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_STD_CXX11)
if (COMPILER_SUPPORTS_STD_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else ()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -march=armv8-a")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES tegra_stereo
  CATKIN_DEPENDS roscpp nodelet
  cv_bridge image_geometry image_proc image_transport
  message_filters sensor_msgs stereo_msgs
  camera_calibration_parsers camera_info_manager
  DEPENDS OpenCV
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
link_directories(${YAML_CPP_LIBRARY_DIRS})

find_package(CUDA REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -w -O3 -gencode arch=compute_53,code=sm_53 -lineinfo")
set(CUDA_ARCH "-arch=sm_53")
set(LIBSGM 
	src/median_filter.cu  
	src/hamming_cost.cu 
	src/disparity_method.cu
	src/debug.cu 
	src/costs.cu
)

cuda_add_library(tegra_stereo_proc src/tegra_stereo_proc.cpp ${LIBSGM})
target_link_libraries(tegra_stereo_proc ${CUDA_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES})


cmake_minimum_required(VERSION 2.8.3)
project(tegra_stereo)

find_package(catkin REQUIRED
  COMPONENTS roscpp nodelet
  cv_bridge image_geometry image_proc image_transport
  message_filters sensor_msgs stereo_msgs
  camera_calibration_parsers camera_info_manager)
find_package(Boost REQUIRED COMPONENTS thread)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_GREATER "3.0.0")
  add_definitions("-DOPENCV3=1")
endif()

find_package(PkgConfig)
pkg_check_modules(YAML_CPP yaml-cpp)
if(${YAML_CPP_VERSION} VERSION_GREATER 0.5)
  add_definitions(-DHAVE_NEW_YAMLCPP)
endif()

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

add_library(tegra_stereo_proc src/tegra_stereo_proc.cpp)
target_link_libraries(tegra_stereo_proc ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${YAML_CPP_LIBRARIES})

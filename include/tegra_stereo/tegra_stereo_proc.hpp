#pragma once

#include <memory>
#include <atomic>
#include <mutex>

#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

#include <image_geometry/stereo_camera_model.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/gpu/gpu.hpp>

#include "tegra_stereo/disparity_method.h"

namespace tegra_stereo {

using stereo_msgs::DisparityImage;
using message_filters::sync_policies::ExactTime;

namespace GPU = cv::gpu;

class TegraStereoProc : public nodelet::Nodelet {
  using SubscriberFilter = image_transport::SubscriberFilter;
  using InfoSubscriber = message_filters::Subscriber<sensor_msgs::CameraInfo>;
  using ExactPolicy = ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;

public:
  TegraStereoProc();
  ~TegraStereoProc();

  virtual void onInit();
  void imageCallback(const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg);

private:
  std::once_flag calibration_initialized_flag;

  boost::shared_ptr<image_transport::ImageTransport> m_it;

  SubscriberFilter m_left_sub, m_right_sub;
  InfoSubscriber m_left_info_sub, m_right_info_sub;

  boost::shared_ptr<ExactSync> m_exact_sync;

  ros::Publisher pub_disparity_; // TODO it_publisher maybe
  
  // stereo pair
  GPU::GpuMat gpu_raw_left_;
  GPU::GpuMat gpu_raw_right_;
  uint32_t l_width_;
  uint32_t l_height_;
  uint32_t r_width_;
  uint32_t r_height_;
  
  // rectification
  image_geometry::PinholeCameraModel left_model_;
  image_geometry::PinholeCameraModel right_model_;

  cv::Mat left_map1_, left_map2_;
  GPU::GpuMat gpu_left_map1_, gpu_left_map2_;
  GPU::GpuMat gpu_left_rect_color_, gpu_left_rect_;

  cv::Mat right_map1_, right_map2_;
  GPU::GpuMat gpu_right_map1_, gpu_right_map2_;
  GPU::GpuMat gpu_right_rect_color_, gpu_right_rect_;

  // stereo matching
  image_geometry::StereoCameraModel stereo_model_;
  int32_t p1_;
  int32_t p2_;

  mutable cv::Mat disparity_;
  GPU::GpuMat gpu_disp_;

  int queue_size_;

  void initRectificationMap(const sensor_msgs::CameraInfoConstPtr &msg,
      cv::Mat& map1, cv::Mat& map2);

  void rectifyMono(GPU::GpuMat& gpu_raw,
      GPU::GpuMat& gpu_rect,
      GPU::GpuMat& gpu_map1,
      GPU::GpuMat& gpu_map2,
      GPU::Stream& stream = GPU::Stream::Null());

  void processStereo(GPU::GpuMat& gpu_left_raw,
      GPU::GpuMat& gpu_right_raw,
      const std_msgs::Header& header);
};

}  // namespace


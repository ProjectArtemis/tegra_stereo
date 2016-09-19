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

#include "tegra_stereo/disparity_method.h"

namespace tegra_stereo {

using stereo_msgs::DisparityImage;
using message_filters::sync_policies::ExactTime;

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

  boost::shared_ptr<image_transport::ImageTransport> it_;

  SubscriberFilter left_raw_sub_, right_raw_sub_;
  InfoSubscriber left_info_sub_, right_info_sub_;

  boost::shared_ptr<ExactSync> exact_sync_;

  ros::Publisher pub_disparity_;
  
  image_transport::Publisher left_rect_pub_;
  image_transport::Publisher right_rect_pub_;
  
  image_transport::Publisher raw_disparity_pub_;
  
  // camera models
  image_geometry::PinholeCameraModel left_model_;
  image_geometry::PinholeCameraModel right_model_;
  image_geometry::StereoCameraModel stereo_model_;

  // stereo matching
  int p1_;
  int p2_;

  int queue_size_;
  
  void publishRectifiedImages(const cv::Mat& left_rect, 
			      const cv::Mat& right_rect, 
			      const sensor_msgs::ImageConstPtr &l_image_msg,
			      const sensor_msgs::ImageConstPtr &r_image_msg);
  void publishDisparity(const cv::Mat& disparity, const std_msgs::Header& header);
  void publishPointcloud(const cv::Mat& disparity);
  
};

}  // namespace


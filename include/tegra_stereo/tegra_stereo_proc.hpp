#ifndef TEGRA_STEREO_TEGRA_STEREO_PROC_HPP_
#define TEGRA_STEREO_TEGRA_STEREO_PROC_HPP_

#include <boost/version.hpp>
#if ((BOOST_VERSION / 100) % 1000) >= 53
#include <boost/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/lock_guard.hpp>
#endif

#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <image_geometry/stereo_camera_model.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

// CUDA
#if OPENCV3
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#define GPU cv::cuda
#else
#include <opencv2/gpu/gpu.hpp>
#define GPU cv::gpu
#endif

#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

using namespace sensor_msgs;
using namespace stereo_msgs;
using namespace message_filters::sync_policies;

namespace tegra_stereo {

class TegraStereoProc : public nodelet::Nodelet {
  typedef image_transport::SubscriberFilter Subscriber;
    typedef message_filters::Subscriber<sensor_msgs::CameraInfo> InfoSubscriber;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;

 public:
  TegraStereoProc();
  ~TegraStereoProc();

  virtual void onInit();
  void imageCallback(const sensor_msgs::ImageConstPtr& l_image_msg,
                 const sensor_msgs::ImageConstPtr& r_image_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg,
                 const sensor_msgs::CameraInfoConstPtr& r_info_msg);
 private:

  bool calibration_initialized;

  boost::shared_ptr<image_transport::ImageTransport> it_;
  
  Subscriber m_left_sub, m_right_sub;
  InfoSubscriber m_left_info_sub, m_right_info_sub;

  boost::shared_ptr<ExactSync> m_exact_sync;

  ros::Publisher pub_disparity_; // TODO it_publisher maybe

  // stereo pair
  GPU::GpuMat gpu_raw_left_;
  GPU::GpuMat gpu_raw_right_;

  // crop
  uint32_t l_x_offset_;
  uint32_t l_y_offset_;
  uint32_t l_width_;
  uint32_t l_height_;
  uint32_t r_x_offset_;
  uint32_t r_y_offset_;
  uint32_t r_width_;
  uint32_t r_height_;

  // rectify
  image_geometry::PinholeCameraModel left_model_;
  image_geometry::PinholeCameraModel right_model_;

  cv::Mat left_map1_, left_map2_;
  GPU::GpuMat gpu_left_map1_, gpu_left_map2_;
  GPU::GpuMat gpu_left_rect_color_, gpu_left_rect_;

  cv::Mat right_map1_, right_map2_;
  GPU::GpuMat gpu_right_map1_, gpu_right_map2_;
  GPU::GpuMat gpu_right_rect_color_, gpu_right_rect_;

  // stretch
  int32_t stretch_factor_;
  GPU::GpuMat gpu_left_stretch_;
  GPU::GpuMat gpu_right_stretch_;

  // stereo matching
  image_geometry::StereoCameraModel stereo_model_;

#if OPENCV3
  mutable cv::Ptr<cv::cuda::StereoBM> block_matcher_;
  mutable cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp_matcher_;
  mutable cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
  mutable cv::cuda::HostMem disparity_;
#else
  mutable cv::gpu::StereoBM_GPU block_matcher_;
  mutable cv::Mat disparity_;
#endif
  
  int queue_size_;
  
  int win_size_;
  int ndisp_;
  int filter_radius_;
  int filter_iter_;
  GPU::GpuMat gpu_disp_, gpu_disp_filtered_, gpu_disp_stretch_;

  // switch
  bool use_csbp_;
  bool use_bilateral_filter_;
  bool use_stretch_;

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

#endif  // TEGRA_STEREO_TEGRA_STEREO_PROC_HPP_

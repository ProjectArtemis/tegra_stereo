#include "tegra_stereo/tegra_stereo_proc.hpp"

namespace tegra_stereo {

TegraStereoProc::TegraStereoProc() {}
TegraStereoProc::~TegraStereoProc() {}

void TegraStereoProc::onInit() {

  ros::NodeHandle &nh = getNodeHandle();
  ros::NodeHandle &private_nh = getPrivateNodeHandle();

  //private_nh.param("ndisparity", ndisp_, 96);
  private_nh.param("P1", p1_, 1000);
  private_nh.param("P2", p2_, 2000);

  private_nh.param("queue_size", queue_size_, 100);
  m_it = boost::make_shared<image_transport::ImageTransport>(nh);
  m_left_sub.subscribe(*m_it.get(), "/stereo/cam0/image_raw", 1);
  m_right_sub.subscribe(*m_it.get(), "/stereo/cam1/image_raw", 1);
  m_left_info_sub.subscribe(nh, "/stereo/cam0/camera_info", 1);
  m_right_info_sub.subscribe(nh, "/stereo/cam1/camera_info", 1);

  // GPU information
  ROS_ASSERT(GPU::getCudaEnabledDeviceCount() > 0);
  GPU::DeviceInfo info(GPU::getDevice()); // TODO : print this

  // Synchronize input topics.
  m_exact_sync = boost::make_shared<ExactSync>(ExactPolicy(queue_size_), m_left_sub,
      m_right_sub, m_left_info_sub,
      m_right_info_sub);
  m_exact_sync->registerCallback(
      boost::bind(&TegraStereoProc::imageCallback, this, _1, _2, _3, _4));

  disparity_.create(cv::Size(l_width_, l_height_), CV_32FC1);

  pub_disparity_ = nh.advertise<DisparityImage>("/stereo/disparity", 1);  // TODO add pointcloud2
  
  init_disparity_method(p1_, p2_);

}

void TegraStereoProc::imageCallback(
    const sensor_msgs::ImageConstPtr &l_image_msg,
    const sensor_msgs::ImageConstPtr &r_image_msg,
    const sensor_msgs::CameraInfoConstPtr &l_info_msg,
    const sensor_msgs::CameraInfoConstPtr &r_info_msg) {

  std::call_once(calibration_initialized_flag, [&, this] () {
    initRectificationMap(l_info_msg, left_map1_, left_map2_);
    gpu_left_map1_.upload(left_map1_);
    gpu_left_map2_.upload(left_map2_);

    initRectificationMap(r_info_msg, right_map1_, right_map2_);
    gpu_right_map1_.upload(right_map1_);
    gpu_right_map2_.upload(right_map2_);

    stereo_model_.fromCameraInfo(l_info_msg, r_info_msg);

    l_width_ = l_info_msg->width;
    l_height_ = l_info_msg->height;
    r_width_ = r_info_msg->width;
    r_height_ = r_info_msg->height;

    NODELET_INFO("Stereo calibration initialized");
  });

  // TODO perf this
  gpu_raw_left_.upload(cv_bridge::toCvShare(l_image_msg, sensor_msgs::image_encodings::MONO8)->image);
  gpu_raw_right_.upload(cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::MONO8)->image);

  processStereo(gpu_raw_left_, gpu_raw_right_, l_info_msg->header);

}

void TegraStereoProc::processStereo(GPU::GpuMat &gpu_left_raw,
    GPU::GpuMat &gpu_right_raw,
    const std_msgs::Header &header) {
    
  GPU::Stream gpu_stream;
  
  // rectify
  rectifyMono(gpu_left_raw, gpu_left_rect_, gpu_left_map1_, gpu_left_map2_,
      gpu_stream);

  rectifyMono(gpu_right_raw, gpu_right_rect_, gpu_right_map1_, gpu_right_map2_,
      gpu_stream);
      
  // TODO publish these, use shared memory if it's faster
  // TODO use streams 
  cv::Mat left_rect;
  gpu_left_rect_.download(left_rect);
  cv::Mat right_rect;
  gpu_right_rect_.download(right_rect);
  
  // Compute
  float elapsed_time_ms;
  disparity_ = compute_disparity_method(left_rect, right_rect, &elapsed_time_ms);
  
  /*
  // static const int DPP = 16; // disparities per pixel
  // ^ means cv::StereoBM has 4 fractional bits,
  // so disparity = value / (2 ^ 4) = value / 16
  // in GPU version, disparity mat has raw disparity
  int DPP = stretch_factor_;  // disparities per pixel
  double inv_dpp = 1.0 / DPP;

  // Allocate new disparity image message
  auto disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header = disp_msg->image.header = header;

  // Compute window of (potentially) valid disparities
  int border = win_size_ / 2;
  int left = ndisp_ + 0 + border - 1;
  int wtf = border + 0;
  int right = l_width_ - 1 - wtf;
  int top = border;
  int bottom = l_height_ - 1 - border;

  disp_msg->valid_window.x_offset = left;
  disp_msg->valid_window.y_offset = top;
  disp_msg->valid_window.width = right - left;
  disp_msg->valid_window.height = bottom - top;

  auto &dimage = disp_msg->image;
  dimage.height = l_height_;
  dimage.width = l_width_;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);

  cv::Mat_<float> dmat(dimage.height, dimage.width, static_cast<float*>(static_cast<void*>(dimage.data.data())),
      dimage.step);

  // Stereo parameters
  disp_msg->f = stereo_model_.right().fx();
  disp_msg->T = stereo_model_.baseline();

  /// @todo Window of (potentially) valid disparities

  // Disparity search range
  disp_msg->min_disparity = 0;
  disp_msg->max_disparity = disp_msg->min_disparity + ndisp_ - 1;
  disp_msg->delta_d = inv_dpp;

#if OPENCV3
  // wait for gpu process completion
  NODELET_INFO(" GPU sync");
  gpu_stream.waitForCompletion();
#endif

  // We convert from fixed-point to float disparity and also adjust for any
  // x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)

#if OPENCV3
  if (use_stretch_) {
    disparity_.createMatHeader().convertTo(
        dmat, dmat.type(), inv_dpp,
        -(stereo_model_.left().cx() -
          stereo_model_.right().cx()));
  } else {
    disparity_.createMatHeader().assignTo(dmat, dmat.type());
  }
#else
  disparity_.convertTo(dmat, dmat.type(), inv_dpp,
      -(stereo_model_.left().cx() -
        stereo_model_.right().cx()));
#endif

  ROS_ASSERT(dmat.data == &dimage.data[0]);
  /// @todo is_bigendian? :)

  if (pub_disparity_.getNumSubscribers() > 0) {
    pub_disparity_.publish(disp_msg);
  }
 */
}

void TegraStereoProc::initRectificationMap(const sensor_msgs::CameraInfoConstPtr &msg,
    cv::Mat &map1, cv::Mat &map2) {

  cv::Mat cv_D(msg->D, true);

  cv::Matx33d cv_K(msg->K.data());
  cv::Matx33d cv_R(msg->R.data());
  cv::Matx34d cv_P(msg->P.data());

  cv::initUndistortRectifyMap(cv_K, cv_D, cv_R, cv_P,
      cv::Size(msg->width, msg->height), CV_32FC1, map1,
      map2);
}

void TegraStereoProc::rectifyMono(GPU::GpuMat &gpu_raw,
    GPU::GpuMat &gpu_rect,
    GPU::GpuMat &gpu_map1,
    GPU::GpuMat &gpu_map2,
    GPU::Stream &stream) {
  GPU::remap(gpu_raw, gpu_rect, gpu_map1, gpu_map2, cv::INTER_LINEAR,
      cv::BORDER_CONSTANT, cv::Scalar(), stream);
}

}  // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(tegra_stereo::TegraStereoProc, nodelet::Nodelet)

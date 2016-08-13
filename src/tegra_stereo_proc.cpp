#include "tegra_stereo/tegra_stereo_proc.hpp"

namespace tegra_stereo {

TegraStereoProc::TegraStereoProc() {}
TegraStereoProc::~TegraStereoProc() {}

void TegraStereoProc::onInit() {

  ROS_WARN("Init nodelet");
  ros::NodeHandle &nh = getNodeHandle();
  ros::NodeHandle &private_nh = getPrivateNodeHandle();

  private_nh.param("ndisparity", ndisp_, 96);
  private_nh.param("win_size", win_size_, 15);
  private_nh.param("filter_radius", filter_radius_, 3);
  private_nh.param("filter_iter", filter_iter_, 1);

  private_nh.param("stretch_factor", stretch_factor_, 2);

  private_nh.param("use_csbp", use_csbp_, true);
  private_nh.param("use_bilateral_filter", use_bilateral_filter_, true);
  private_nh.param("use_stretch", use_stretch_, false);

  if (!use_stretch_) {
    stretch_factor_ = 1;
  }

  private_nh.param("queue_size", queue_size_, 100);

  m_it = boost::make_shared<image_transport::ImageTransport>(nh);
  m_left_sub.subscribe(m_it, "/stereo/cam0/image_raw", 1);
  m_right_sub.subscribe(m_it, "/stereo/cam1/image_raw", 1);
  m_left_info_sub.subscribe(nh, "/stereo/cam0/camera_info", 1);
  m_right_info_sub.subscribe(nh, "/stereo/cam1/camera_info", 1);

  // GPU information
  ROS_ASSERT(GPU::getCudaEnabledDeviceCount() > 0);
  GPU::DeviceInfo info(GPU::getDevice());

  // Synchronize input topics.
  m_exact_sync = boost::make_shared<ExactSync>(ExactPolicy(queue_size_), m_left_sub,
      m_right_sub, m_left_info_sub,
      m_right_info_sub);
  m_exact_sync->registerCallback(
      boost::bind(&TegraStereoProc::imageCallback, this, _1, _2, _3, _4));

  disparity_.create(cv::Size(l_width_, l_height_), CV_32FC1);

#if OPENCV3
  block_matcher_ = cv::cuda::createStereoBM(ndisp_ * stretch_factor_,
      win_size_);
  csbp_matcher_ =
    cv::cuda::createStereoConstantSpaceBP(ndisp_ * stretch_factor_,
        8, 4, 4, CV_16SC1);
  //block_matcher_->setPrefilterType(cv::StereoBM::PREFILTER_XSOBEL);
  //block_matcher_->setPrefilterCap(31);
  //block_matcher_->setTextureTheshold(10);
  bilateral_filter_ =
    cv::cuda::createDisparityBilateralFilter(ndisp_ * stretch_factor_,
        filter_radius_, filter_iter_);
#else
  block_matcher_.preset = cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL;
  block_matcher_.ndisp = ndisp_;
  block_matcher_.winSize = win_size_;
  block_matcher_.avergeTexThreshold = 10.0;
#endif

  //ros::SubscriberStatusCallback connect_cb =
  //    boost::bind(&TegraStereoProc::connectCallback, this);
  //image_transport::SubscriberStatusCallback connect_cb_image =
  //    boost::bind(&TegraStereoProc::connectCallback, this);

  pub_disparity_ = nh.advertise<DisparityImage>("/stereo/disparity", 1);  // TODO add pointcloud2
}

void TegraStereoProc::imageCallback(
    const sensor_msgs::ImageConstPtr &l_image_msg,
    const sensor_msgs::ImageConstPtr &r_image_msg,
    const sensor_msgs::CameraInfoConstPtr &l_info_msg,
    const sensor_msgs::CameraInfoConstPtr &r_info_msg) {

  ROS_INFO("image callback");
  if (!calibration_initialized) {
    ROS_INFO("calib init");

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

    calibration_initialized = true;
    ROS_INFO("stereo calibration initialized");
  }

  // TODO perf this
  gpu_raw_left_.upload(cv_bridge::toCvShare(l_image_msg, sensor_msgs::image_encodings::BGR8)->image);
  gpu_raw_right_.upload(cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::BGR8)->image);

  processStereo(gpu_raw_left_, gpu_raw_right_, l_info_msg->header);

  ROS_INFO("end callback");
}

void TegraStereoProc::processStereo(GPU::GpuMat &gpu_left_raw,
    GPU::GpuMat &gpu_right_raw,
    const std_msgs::Header &header) {
  ROS_INFO("start proc");
  GPU::Stream gpu_stream;

  // TODO potentially use this to block out unmatchable portions
  //  // crop left and right images
  //  ROS_INFO("  crop left");
  //  GPU::GpuMat gpu_left_raw(
  //      gpu_input,
  //      cv::Rect(l_x_offset_, l_y_offset_, l_width_, l_height_));

  //  ROS_INFO("  crop right");
  //  GPU::GpuMat gpu_right_raw(
  //      gpu_input,
  //      cv::Rect(r_x_offset_, r_y_offset_, r_width_, r_height_));

  // rectify
  ROS_INFO("  rectify left");
  rectifyMono(gpu_left_raw, gpu_left_rect_, gpu_left_map1_, gpu_left_map2_,
      gpu_stream);

  ROS_INFO("  rectify right");
  rectifyMono(gpu_right_raw, gpu_right_rect_, gpu_right_map1_, gpu_right_map2_,
      gpu_stream);

  // stretch
  if (use_stretch_) {
    ROS_INFO("  stretch left");
    GPU::resize(gpu_left_rect_, gpu_left_stretch_,
        cv::Size(stretch_factor_ * l_width_, l_height_), 0, 0,
        cv::INTER_LINEAR, gpu_stream);

    ROS_INFO("  stretch right");
    GPU::resize(gpu_right_rect_, gpu_right_stretch_,
        cv::Size(stretch_factor_ * r_width_, r_height_), 0, 0,
        cv::INTER_LINEAR, gpu_stream);
  }

  // wait for left and right image
  // gpu_stream.waitForCompletion();

  ROS_INFO(" stereo matching");
  ROS_INFO(" stereoBM in");

#if OPENCV3
  if (use_stretch_) {
    if (use_csbp_) {
      csbp_matcher_->compute(gpu_left_stretch_, gpu_right_stretch_,
          gpu_disp_stretch_, gpu_stream);
    } else {
      block_matcher_->compute(gpu_left_stretch_, gpu_right_stretch_,
          gpu_disp_stretch_, gpu_stream);
    }
    if (use_bilateral_filter_) {
      bilateral_filter_->apply(gpu_disp_stretch_, gpu_left_stretch_,
          gpu_disp_filtered_, gpu_stream);
      GPU::resize(gpu_disp_filtered_, gpu_disp_,
          cv::Size(l_width_, l_height_), 0, 0, cv::INTER_LINEAR,
          gpu_stream);
      gpu_disp_filtered_.download(disparity_, gpu_stream);
    } else {
      GPU::resize(gpu_disp_stretch_, gpu_disp_,
          cv::Size(l_width_, l_height_), 0, 0, cv::INTER_LINEAR,
          gpu_stream);
      gpu_disp_.download(disparity_, gpu_stream);
    }
  } else {
    if (use_csbp_) {
      csbp_matcher_->compute(gpu_left_rect_, gpu_right_rect_, gpu_disp_,
          gpu_stream);
    } else {
      block_matcher_->compute(gpu_left_rect_, gpu_right_rect_, gpu_disp_,
          gpu_stream);
    }
    if (use_bilateral_filter_) {
      bilateral_filter_->apply(gpu_disp_, gpu_left_rect_, gpu_disp_filtered_,
          gpu_stream);
      gpu_disp_filtered_.download(disparity_, gpu_stream);
    } else {
      gpu_disp_.download(disparity_, gpu_stream);
    }
  }

#else
  // TODO add filters, etc.
  block_matcher_(gpu_left_rect_, gpu_right_rect_, gpu_disp_, gpu_stream);
  gpu_stream.enqueueDownload(gpu_disp_, disparity_);
#endif

  ROS_INFO("  stereoBM out");

  // static const int DPP = 16; // disparities per pixel
  // ^ means cv::StereoBM has 4 fractional bits,
  // so disparity = value / (2 ^ 4) = value / 16
  // in GPU version, disparity mat has raw disparity
  int DPP = stretch_factor_;  // disparities per pixel
  double inv_dpp = 1.0 / DPP;

  // Allocate new disparity image message
  DisparityImagePtr disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header = header;
  disp_msg->image.header = header;

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

  sensor_msgs::Image &dimage = disp_msg->image;
  dimage.height = l_height_;
  dimage.width = l_width_;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);
  cv::Mat_<float> dmat(dimage.height, dimage.width, (float *)&dimage.data[0],
      dimage.step);

  // Stereo parameters
  disp_msg->f = stereo_model_.right().fx();
  disp_msg->T = stereo_model_.baseline();

  /// @todo Window of (potentially) valid disparities

  // Disparity search range
  disp_msg->min_disparity = 0;
  disp_msg->max_disparity = disp_msg->min_disparity + ndisp_ - 1;
  disp_msg->delta_d = inv_dpp;

  // wait for gpu process completion
  ROS_INFO(" GPU sync");
  gpu_stream.waitForCompletion();
  ROS_INFO(" GPU process end");

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
  ROS_INFO(" disparity conversion end");

  if (pub_disparity_.getNumSubscribers() > 0) {
    pub_disparity_.publish(disp_msg);
  }

  ROS_INFO("end proc");
}

void TegraStereoProc::initRectificationMap(const sensor_msgs::CameraInfoConstPtr &msg,
    cv::Mat &map1, cv::Mat &map2) {
  cv::Mat_<double> cv_D(1, msg->D.size());
  for (size_t i = 0; i < msg->D.size(); i++) {
    cv_D(i) = msg->D[i];
  }

  cv::Matx33d cv_K(&msg->K[0]);
  cv::Matx33d cv_R(&msg->R[0]);
  cv::Matx34d cv_P(&msg->P[0]);

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

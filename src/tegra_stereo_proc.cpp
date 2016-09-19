/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tegra_stereo/tegra_stereo_proc.hpp"

namespace tegra_stereo {

TegraStereoProc::TegraStereoProc() {}
TegraStereoProc::~TegraStereoProc() {}

void TegraStereoProc::onInit() {

  ros::NodeHandle &nh = getNodeHandle();
  ros::NodeHandle &private_nh = getPrivateNodeHandle();

  private_nh.param("P1", p1_, 20);
  private_nh.param("P2", p2_, 100);

  private_nh.param("queue_size", queue_size_, 100);
  
  it_ = boost::make_shared<image_transport::ImageTransport>(nh);
  
  left_raw_sub_.subscribe(*it_.get(), "/stereo/cam0/image_raw", 1);
  right_raw_sub_.subscribe(*it_.get(), "/stereo/cam1/image_raw", 1);
  left_info_sub_.subscribe(nh, "/stereo/cam0/camera_info", 1);
  right_info_sub_.subscribe(nh, "/stereo/cam1/camera_info", 1);
  
  left_rect_pub_ = it_->advertise("/stereo/cam0/image_rect", 1);
  right_rect_pub_ = it_->advertise("/stereo/cam1/image_rect", 1);
  raw_disparity_pub_ = it_->advertise("/stereo/disparity_raw", 1);
  pub_disparity_ = nh.advertise<DisparityImage>("/stereo/disparity", 1);
  
  // Synchronize input topics
  exact_sync_ = boost::make_shared<ExactSync>(ExactPolicy(queue_size_), left_raw_sub_,
      right_raw_sub_, left_info_sub_,
      right_info_sub_);
  exact_sync_->registerCallback(
      boost::bind(&TegraStereoProc::imageCallback, this, _1, _2, _3, _4));
  
  // Initialize Semi-Global Matcher
  init_disparity_method(p1_, p2_);

}

void TegraStereoProc::imageCallback(
    const sensor_msgs::ImageConstPtr &l_image_msg,
    const sensor_msgs::ImageConstPtr &r_image_msg,
    const sensor_msgs::CameraInfoConstPtr &l_info_msg,
    const sensor_msgs::CameraInfoConstPtr &r_info_msg) {

  std::call_once(calibration_initialized_flag, [&, this] () {
  
    left_model_.fromCameraInfo(l_info_msg);
    right_model_.fromCameraInfo(r_info_msg);
    stereo_model_.fromCameraInfo(l_info_msg, r_info_msg);

    NODELET_INFO("Stereo calibration initialized");
    
  });
  
  const cv::Mat left_raw = cv_bridge::toCvShare(l_image_msg, sensor_msgs::image_encodings::MONO8)->image;
  const cv::Mat right_raw = cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::MONO8)->image;
  cv::Mat left_rect;
  cv::Mat right_rect;
  
  left_model_.rectifyImage(left_raw, left_rect, cv::INTER_LINEAR);
  right_model_.rectifyImage(right_raw, right_rect, cv::INTER_LINEAR);
  
  // Compute
  float elapsed_time_ms;
  cv::Mat disparity = compute_disparity_method(left_raw, right_raw, &elapsed_time_ms);
  
  NODELET_INFO(" Disparity computation took %f ms", elapsed_time_ms);
  
  publishRectifiedImages(left_rect, right_rect, l_image_msg, r_image_msg);
  publishDisparity(disparity, l_info_msg->header);

}

void TegraStereoProc::publishRectifiedImages(const cv::Mat& left_rect, 
					     const cv::Mat& right_rect, 
					     const sensor_msgs::ImageConstPtr &l_image_msg,
					     const sensor_msgs::ImageConstPtr &r_image_msg)
{

  sensor_msgs::ImagePtr left_rect_msg = cv_bridge::CvImage(l_image_msg->header, l_image_msg->encoding, left_rect).toImageMsg();
  sensor_msgs::ImagePtr right_rect_msg = cv_bridge::CvImage(r_image_msg->header, r_image_msg->encoding, right_rect).toImageMsg();
  left_rect_pub_.publish(left_rect_msg);
  right_rect_pub_.publish(right_rect_msg);
}

void TegraStereoProc::publishDisparity(const cv::Mat& disparity, const std_msgs::Header& header)
{

  // publish raw disparity from algorithm
  sensor_msgs::ImagePtr raw_disp_msg = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, disparity).toImageMsg();
  raw_disparity_pub_.publish(raw_disp_msg);
  
  // Publish disparity
  static const int DPP = 1; // disparities per pixel
  static const double inv_dpp = 1.0 / DPP;
  
  auto disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header = disp_msg->image.header = header;

  auto &dimage = disp_msg->image;
  dimage.height = 512; // TODO hack
  dimage.width = 640;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);

  const cv::Mat_<float> dmat(dimage.height, dimage.width, static_cast<float*>(static_cast<void*>(dimage.data.data())),
      dimage.step);
  // TODO check why these are different
  //const cv::Mat_<float> dmat(dimage.height, dimage.width, (float*)&dimage.data[0], dimage.step);

  // Stereo parameters
  disp_msg->f = stereo_model_.right().fx();
  disp_msg->T = stereo_model_.baseline();

  // Disparity search range
  disp_msg->min_disparity = 0;
  disp_msg->max_disparity = 127;
  disp_msg->delta_d = inv_dpp;

  // We convert from fixed-point to float disparity and also adjust for any
  // x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
  disparity.convertTo(dmat, dmat.type(), inv_dpp,
      -(stereo_model_.left().cx() -
        stereo_model_.right().cx()));

  ROS_ASSERT(dmat.data == &dimage.data[0]);

  if (pub_disparity_.getNumSubscribers() > 0) {
    pub_disparity_.publish(disp_msg);
  }

}

void TegraStereoProc::publishPointcloud(const cv::Mat& disparity)
{
  // Calculate dense point cloud
  /*const sensor_msgs::Image& dimage = disparity.image;
  
  const cv::Mat_<float> dmat(dimage.height, dimage.width, (float*)&dimage.data[0], dimage.step);
  model.projectDisparityImageTo3d(dmat, dense_points_, true);

  // Fill in sparse point cloud message
  points.height = dense_points_.rows;
  points.width  = dense_points_.cols;
  points.fields.resize (4);
  points.fields[0].name = "x";
  points.fields[0].offset = 0;
  points.fields[0].count = 1;
  points.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  points.fields[1].name = "y";
  points.fields[1].offset = 4;
  points.fields[1].count = 1;
  points.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  points.fields[2].name = "z";
  points.fields[2].offset = 8;
  points.fields[2].count = 1;
  points.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  points.fields[3].name = "rgb";
  points.fields[3].offset = 12;
  points.fields[3].count = 1;
  points.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
  //points.is_bigendian = false; ???
  points.point_step = 16; // TODO : why?
  points.row_step = points.point_step * points.width;
  points.data.resize (points.row_step * points.height);
  points.is_dense = false; // there may be invalid points
 
  float bad_point = std::numeric_limits<float>::quiet_NaN ();
  int i = 0;
  for (int32_t u = 0; u < dense_points_.rows; ++u) {
    for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
      if (isValidPoint(dense_points_(u,v))) {
        // x,y,z,rgba
        memcpy (&points.data[i * points.point_step + 0], &dense_points_(u,v)[0], sizeof (float));
        memcpy (&points.data[i * points.point_step + 4], &dense_points_(u,v)[1], sizeof (float));
        memcpy (&points.data[i * points.point_step + 8], &dense_points_(u,v)[2], sizeof (float));
      }
      else {
        memcpy (&points.data[i * points.point_step + 0], &bad_point, sizeof (float));
        memcpy (&points.data[i * points.point_step + 4], &bad_point, sizeof (float));
        memcpy (&points.data[i * points.point_step + 8], &bad_point, sizeof (float));
      }
    }
  }

  // Fill in color
  namespace enc = sensor_msgs::image_encodings;
  i = 0;
  if (encoding == enc::MONO8) {
    for (int32_t u = 0; u < dense_points_.rows; ++u) {
      for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
        if (isValidPoint(dense_points_(u,v))) {
          uint8_t g = color.at<uint8_t>(u,v);
          int32_t rgb = (g << 16) | (g << 8) | g;
          memcpy (&points.data[i * points.point_step + 12], &rgb, sizeof (int32_t));
        }
        else {
          memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
        }
      }
    }
  }
  else if (encoding == enc::RGB8) {
    for (int32_t u = 0; u < dense_points_.rows; ++u) {
      for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
        if (isValidPoint(dense_points_(u,v))) {
          const cv::Vec3b& rgb = color.at<cv::Vec3b>(u,v);
          int32_t rgb_packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
          memcpy (&points.data[i * points.point_step + 12], &rgb_packed, sizeof (int32_t));
        }
        else {
          memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
        }
      }
    }
  }
  else if (encoding == enc::BGR8) {
    for (int32_t u = 0; u < dense_points_.rows; ++u) {
      for (int32_t v = 0; v < dense_points_.cols; ++v, ++i) {
        if (isValidPoint(dense_points_(u,v))) {
          const cv::Vec3b& bgr = color.at<cv::Vec3b>(u,v);
          int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
          memcpy (&points.data[i * points.point_step + 12], &rgb_packed, sizeof (int32_t));
        }
        else {
          memcpy (&points.data[i * points.point_step + 12], &bad_point, sizeof (float));
        }
      }
    }
  }
  else {
    ROS_WARN("Could not fill color channel of the point cloud, unrecognized encoding '%s'", encoding.c_str());
  }
  
  */
}


}  // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(tegra_stereo::TegraStereoProc, nodelet::Nodelet)

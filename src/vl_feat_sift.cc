

// Glog
#include <glog/logging.h>

// Original
#include "vl_feat_sift.h"

namespace {
double GetValidFirstOctave(const int first_octave, const int width, const int height,
                           const double max_scaled_dim) {
  const int max_dim = std::max(width, height);
  int valid_first_octave = first_octave;
  double scale_factor = std::pow(2.0, -1 * valid_first_octave);
  while (max_dim * scale_factor >= max_scaled_dim) {
    scale_factor /= 2.0;
    ++valid_first_octave;
  }
  return valid_first_octave;
}
}  // namespace

void ComputeKeyPointsAndDescriptors(const cv::Mat& gray_image, const SIFTParams& params,
                                    std::vector<KeyPoint>& keypoints,
                                    std::vector<Eigen::VectorXf>& descriptors) {
  keypoints.clear();
  descriptors.clear();

  // 0. Convert image to float.
  cv::Mat float_img(gray_image.rows, gray_image.cols, CV_32FC1);
  gray_image.convertTo(float_img, CV_32FC1);

  // 1. Compute Valid First Octave.
  const int first_octave = GetValidFirstOctave(params.def_first_octave_, float_img.cols,
                                               float_img.rows, params.max_scaled_dim_);

  // 2. Create new sift filter.
  VlSiftFilt* sift_filter = vl_sift_new(float_img.cols, float_img.rows, params.num_octave_,
                                        params.num_levels_, first_octave);

  // 3. Set Parameters.
  vl_sift_set_peak_thresh(sift_filter, params.peak_threshold_ / 255.0);
  vl_sift_set_edge_thresh(sift_filter, params.edge_threshold_ / 255.0);
  vl_sift_set_norm_thresh(sift_filter, params.norm_threshold_ / 255.0);
  vl_sift_set_magnif(sift_filter, params.magnif_);
  vl_sift_set_window_size(sift_filter, params.window_size_);

  // 4. Process First Octave.
  int octave_no = 1;
  int vl_status =
      vl_sift_process_first_octave(sift_filter, reinterpret_cast<float*>(float_img.data));

  // 5. Process Octave until we cant anymore.
  while (vl_status != VL_ERR_EOF) {
    // Detect Key Points.
    vl_sift_detect(sift_filter);

    // Get Key Points.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift_filter);
    const int num_keypoints = vl_sift_get_nkeypoints(sift_filter);

    /*
    LOG(INFO) << "Processing Octave : " << octave_no;
    LOG(INFO) << "Octave Width : " << vl_sift_get_octave_width(sift_filter);
    LOG(INFO) << "Octave Height : " << vl_sift_get_octave_height(sift_filter);
    LOG(INFO) << "Num Keypoints : " << num_keypoints;
    */

    // Calc Orientation for keypoints detected.
    for (int i = 0; i < num_keypoints; ++i) {
      // Calculate orientations of the keypoint.
      double angles[4];
      int num_angles = vl_sift_calc_keypoint_orientations(sift_filter, angles, &vl_keypoints[i]);

      // Compute for Each Key Point.
      Eigen::VectorXf descriptor(params.num_sift_dimensions_);
      for (int j = 0; j < num_angles; ++j) {
        descriptor.setZero();
        vl_sift_calc_keypoint_descriptor(sift_filter, descriptor.data(), &vl_keypoints[i],
                                         angles[j]);
        descriptors.push_back(descriptor);
        KeyPoint kp(vl_keypoints[i].x, vl_keypoints[i].y, vl_keypoints[i].sigma, angles[j], 0,
                    vl_keypoints[i].o);
        keypoints.push_back(kp);
      }
    }
    vl_status = vl_sift_process_next_octave(sift_filter);
    octave_no = octave_no + 1;
  }

  vl_sift_delete(sift_filter);
}
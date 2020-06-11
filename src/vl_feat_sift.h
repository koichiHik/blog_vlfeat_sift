
#ifndef _VL_FEAT_SIFT_H_
#define _VL_FEAT_SIFT_H_

// STL
#include <memory>

#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>

// VLFEAT Library.
extern "C" {
#include "extern/lib/include/vl/sift.h"
}

// Original
#include "keypoints.h"

struct SIFTParams {
  SIFTParams(int max_scaled_dim, int num_sift_dimensions, int num_octave, int num_levels,
             int def_first_octave, double peak_threshold, double edge_threshold,
             double norm_threshold, double magnif, double window_size, bool root_sift,
             bool upright_sift)
      : max_scaled_dim_(max_scaled_dim),
        num_sift_dimensions_(num_sift_dimensions),
        num_octave_(num_octave),
        num_levels_(num_levels),
        def_first_octave_(def_first_octave),
        peak_threshold_(peak_threshold),
        edge_threshold_(edge_threshold),
        norm_threshold_(norm_threshold),
        magnif_(magnif),
        window_size_(window_size),
        root_sift_(root_sift),
        upright_sift_(upright_sift) {}

  int max_scaled_dim_;
  int num_sift_dimensions_;

  int num_octave_;
  int num_levels_;
  int def_first_octave_;

  double peak_threshold_;
  double edge_threshold_;
  double norm_threshold_;
  double magnif_;
  double window_size_;

  bool root_sift_;
  bool upright_sift_;

 public:
  friend std::ostream& operator<<(std::ostream& output, const SIFTParams& param) {
    output << param.max_scaled_dim_ << ", " << param.num_sift_dimensions_ << ", "
           << param.num_octave_ << ", " << param.num_levels_ << ", " << param.def_first_octave_
           << ", " << param.peak_threshold_ << ", " << param.edge_threshold_ << ", "
           << param.norm_threshold_ << ", " << param.magnif_ << ", " << param.window_size_;
    return output;
  }
};

void ComputeKeyPointsAndDescriptors(const cv::Mat& gray_image, const SIFTParams& params,
                                    std::vector<KeyPoint>& keypoints,
                                    std::vector<Eigen::VectorXf>& descriptors);

#endif
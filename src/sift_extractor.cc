
// STL
#include <filesystem>
#include <fstream>
#include <string>

// Glog
#include <glog/logging.h>

// GFlag
#include <gflags/gflags.h>

// Cereal
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>

// Eigen
#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Original
#include "eigen_serializable.h"
#include "vl_feat_sift.h"

DEFINE_string(data_directory, "", "");

// Maximum scaled dimension to extract descriptors. If the dimension is larger
// than this then we begin to have memory and speed issues.
// CONST Parameters.
DEFINE_int32(kMaxScaledDim, 3600, "");
DEFINE_int32(kNumSiftDimensions, 128, "");

// SIFT Parameters
DEFINE_int32(num_octaves, -1, "");
DEFINE_int32(num_levels, 3, "");
DEFINE_int32(def_first_octave, -1, "");
DEFINE_int32(max_num_feature, 10000, "");

DEFINE_double(peak_threshold, 1.2f, "");
DEFINE_double(edge_threshold, 10.0f, "");
DEFINE_double(norm_threshold, 0.0f, "");
DEFINE_double(magnif, 3.0f, "");
DEFINE_double(window_size, 2.0f, "");

DEFINE_bool(root_sift, false, "");
DEFINE_bool(upright_sift, false, "");

// Value defined in CMakeLists.txt file.
static const std::string project_folder_path = PRJ_FOLDER_PATH;

void ExtractAllFilePathsInDirectory(const std::string& dir_path,
                                    std::vector<std::string>& filenames) {
  namespace fs = std::filesystem;
  for (const fs::directory_entry& file : fs::directory_iterator(dir_path)) {
    if (!file.is_directory()) {
      filenames.push_back(file.path());
    }
  }
  LOG(INFO) << "Count of found files : " << filenames.size();
}

std::filesystem::path CreateSavingDirectory(const std::string& src_filepath) {
  const std::filesystem::path rep_path(src_filepath);
  const std::filesystem::path parent_dir = rep_path.parent_path().replace_filename("sift");
  std::filesystem::create_directories(parent_dir);
  return parent_dir;
}

std::filesystem::path CreateFileName(const std::string& src_filepath) {
  std::filesystem::path org_path(src_filepath);
  return org_path.replace_extension(".bin").filename();
}

void ComputeSiftFeatures(const std::vector<std::string>& filepaths) {
  SIFTParams params(FLAGS_kMaxScaledDim, FLAGS_kNumSiftDimensions, FLAGS_num_octaves,
                    FLAGS_num_levels, FLAGS_def_first_octave, FLAGS_peak_threshold,
                    FLAGS_edge_threshold, FLAGS_norm_threshold, FLAGS_magnif, FLAGS_window_size,
                    FLAGS_root_sift, FLAGS_upright_sift);

  CHECK(filepaths.size() > 0);

  // X. Create saving directory.
  const std::filesystem::path dir_path(CreateSavingDirectory(filepaths[0]));

  // X. Compute SIFT and save.
  int size = filepaths.size();
  for (int idx = 0; idx < size; idx++) {
    const std::string image_path = filepaths[idx];
    std::vector<KeyPoint> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    cv::Mat gray_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    // X. Compute SIFT keypoints and descriptors.
    LOG(INFO) << "Compute SIFT features.... Current status : " << idx << " / " << size;
    ComputeKeyPointsAndDescriptors(gray_image, params, keypoints, descriptors);
    LOG(INFO) << "SIFT features computed! Key Point Count : " << keypoints.size();

    if (FLAGS_max_num_feature <= keypoints.size()) {
      keypoints.resize(FLAGS_max_num_feature);
      descriptors.resize(FLAGS_max_num_feature);
    }

    // X. Serialize data and save.
    {
      std::string bin_path =
          std::filesystem::path(dir_path).append(CreateFileName(image_path).string());
      LOG(INFO) << "Saving binary to : " << bin_path;
      std::ofstream feature_writer(bin_path, std::ios::out | std::ios::binary);
      cereal::PortableBinaryOutputArchive output_archive(feature_writer);
      output_archive(keypoints, descriptors);
    }
  }
}

int main(int argc, char** argv) {
  // X. Initial setting.
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);
  const std::string dir_path =
      FLAGS_data_directory.empty() ? project_folder_path + "/data/images" : FLAGS_data_directory;

  // X. Show data directory.
  LOG(INFO) << "Directory path to be searched : " << dir_path;

  // X. Extract all file names.
  std::vector<std::string> filenames;
  ExtractAllFilePathsInDirectory(dir_path, filenames);

  // X. Compute sift features.
  ComputeSiftFeatures(filenames);

  return 0;
}

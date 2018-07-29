#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fs {
// Returns paths to all images in given directory - they should be sorted
// starting from the closes one to camera
std::vector<std::string> GetImagesPaths(const std::string& directory_path);

//Loads the images from provided directory
std::unique_ptr<std::vector<cv::Mat>> LoadImages(const std::string& directory_path);
}  // namespace fs

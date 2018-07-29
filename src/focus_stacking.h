#pragma once

#include <opencv2/opencv.hpp>

namespace fs {
  void GaussianBlur(cv::Mat* img);
  cv::Mat ComputeWeights(const cv::Mat& img, const cv::Mat& kernel,
                                int channel);
  cv::Mat Laplacian(const cv::Mat& img, int channel);
}  // namespace fs

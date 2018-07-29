#pragma once

#include "image_loader.h"
#include <opencv2/opencv.hpp>

namespace fs {
enum class SelectedChannel { BLUE, GREEN, RED };

class FocusStacking {
  public:
    FocusStacking(std::string images_directory,
                  SelectedChannel selected_channel)
        : images_directory_(images_directory),
          selected_channel_(selected_channel) {
      images_ = fs::LoadImages(images_directory_);
    }


    cv::Mat ComputeSharpImage();

    static void GaussianBlur(cv::Mat *img);
    // Computes weights which indicate if given pixel belong to some edge
    static cv::Mat ComputeWeights(const cv::Mat &img, const cv::Mat &kernel,
                           int channel);
    // Perform Laplacian and return weights
    static cv::Mat Laplacian(const cv::Mat &img, int channel);

  private:
    std::string images_directory_;
    SelectedChannel selected_channel_;
    std::unique_ptr<std::vector<cv::Mat>> images_;


    std::vector<cv::Mat> ProcessImages(const std::vector<cv::Mat>& images);
};
}  // namespace fs

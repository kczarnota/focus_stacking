#pragma once

#include "image_loader.h"
#include <opencv2/opencv.hpp>

namespace fs {
enum class SelectedChannel { BLUE, GREEN, RED };
enum class EdgeDetectionMethod { LAPLACIAN, SOBEL };

class FocusStacking {
  public:
    FocusStacking(std::string images_directory,
                  EdgeDetectionMethod edge_detection_method,
                  SelectedChannel selected_channel, int edge_threshold,
                  int not_defined_depth_margin)
        : images_directory_(images_directory),
          edge_detection_method_(edge_detection_method),
          selected_channel_(selected_channel),
          edge_threshold_(edge_threshold),
          not_defined_depth_margin_(not_defined_depth_margin) {
      images_ = fs::LoadImages(images_directory_);
    }

    explicit FocusStacking(std::string images_directory)
        : FocusStacking(images_directory, EdgeDetectionMethod::SOBEL,
                        SelectedChannel::GREEN, 30, 40) {}


    std::pair<cv::Mat, cv::Mat> ComputeSharpImageAndDepthMap();

    static void GaussianBlur(cv::Mat *img);
    // Computes weights which indicate if given pixel belong to some edge
    static cv::Mat ComputeWeights(const cv::Mat &img, const cv::Mat &kernel,
                           int channel);
    // Perform Laplacian and return weights
    static cv::Mat Laplacian(const cv::Mat &img, int channel);
    static cv::Mat Sobel(const cv::Mat& img, int channel);
    static std::vector<uchar> PrepareLookupTableWithColors(
      size_t number_of_images, uchar edge_threshold);

  private:
    static const uchar kDephtColorMaxWalue_ = 255;

    std::string images_directory_;
    EdgeDetectionMethod edge_detection_method_;
    SelectedChannel selected_channel_;
    int edge_threshold_;
    int not_defined_depth_margin_;
    std::unique_ptr<std::vector<cv::Mat>> images_;


    std::vector<cv::Mat> ProcessImages(const std::vector<cv::Mat>& images);
};
}  // namespace fs

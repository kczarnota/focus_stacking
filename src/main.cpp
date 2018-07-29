#include "focus_stacking.h"
#include "image_loader.h"
#include <opencv2/opencv.hpp>
#include <iostream>


void PerformTests(std::string images_directory_path);

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    // Directory should contain only images, sorted from the closest one
    std::cout << "Usage: focus_stacking <path_to_directory_with_images>" << std::endl;
    exit(-1);
  }

  fs::FocusStacking focus_stacking(argv[1]);
  auto results = focus_stacking.ComputeSharpImageAndDepthMap();
  cv::imwrite("result.png", results.first);
  cv::imwrite("depth.png", results.second);

  return 0;
}

void PerformTests(std::string images_directory_path) {
  fs::FocusStacking focus_stacking(images_directory_path);
  std::vector<fs::SelectedChannel> channels = {fs::SelectedChannel::RED,
                                               fs::SelectedChannel::GREEN,
                                               fs::SelectedChannel::BLUE};
  std::vector<fs::EdgeDetectionMethod> methods = {fs::EdgeDetectionMethod::LAPLACIAN,
                                                  fs::EdgeDetectionMethod::SOBEL};
  std::vector<double> edge_thresholds = {10, 20, 30};
  std::vector<uchar> margins = {30, 40, 50};
  focus_stacking.PerformTests(channels, methods, edge_thresholds, margins);
}

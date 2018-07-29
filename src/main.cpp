#include "focus_stacking.h"
#include "image_loader.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    // Directory should contain only images, sorted from the closest one
    std::cout << "Usage: focus_stacking <path_to_directory_with_images>" << std::endl;
    exit(-1);
  }

  fs::FocusStacking focus_stacking(argv[1], fs::SelectedChannel::GREEN);
  auto result = focus_stacking.ComputeSharpImageAndDepthMap();
  //cv::imshow("img", result.first);
  cv::imwrite("first_try.png", result.first);
  cv::imwrite("depth_first_try.png", result.second);
  cv::waitKey();
  return 0;
}

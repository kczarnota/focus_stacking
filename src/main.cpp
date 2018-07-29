#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
  cv::Mat img = cv::imread("bug/result.png");
  cv::imshow("img", img);
  cv::waitKey();
  return 0;
}

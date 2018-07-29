#include "focus_stacking.h"

void fs::GaussianBlur(cv::Mat* img) {
  cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
  int sum_of_weights = 0;

  for (int r = 0; r < kernel.rows; ++r) {
    for (int c = 0; c < kernel.cols; ++c) {
      sum_of_weights += kernel.at<uchar>(r, c);
    }
  }

  cv::Mat img_copy = img->clone();
  for (int r = 1; r < img->rows - 1; ++r) {
    for (int c = 1; c < img->cols - 1; ++c) {
      for (int d = 0; d < 3; ++d) {
        int sum =
            kernel.at<uchar>(0, 0) * img_copy.at<cv::Vec3b>(r - 1, c - 1)[d] +
            kernel.at<uchar>(0, 1) * img_copy.at<cv::Vec3b>(r - 1, c)[d] +
            kernel.at<uchar>(0, 2) * img_copy.at<cv::Vec3b>(r - 1, c + 1)[d] +
            kernel.at<uchar>(1, 0) * img_copy.at<cv::Vec3b>(r, c - 1)[d] +
            kernel.at<uchar>(1, 1) * img_copy.at<cv::Vec3b>(r, c)[d] +
            kernel.at<uchar>(1, 2) * img_copy.at<cv::Vec3b>(r, c + 1)[d] +
            kernel.at<uchar>(2, 0) * img_copy.at<cv::Vec3b>(r + 1, c - 1)[d] +
            kernel.at<uchar>(2, 1) * img_copy.at<cv::Vec3b>(r + 1, c)[d] +
            kernel.at<uchar>(2, 2) * img_copy.at<cv::Vec3b>(r + 1, c + 1)[d];

        sum /= sum_of_weights;
        img->at<cv::Vec3b>(r, c)[d] = sum;
      }
    }
  }
}

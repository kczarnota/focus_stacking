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


cv::Mat fs::ComputeWeights(const cv::Mat& img,
                                          const cv::Mat& kernel, int channel) {
  cv::Mat pixel_weights = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);

  for (int r = 1; r < img.rows - 1; ++r) {
    for (int c = 1; c < img.cols - 1; ++c) {
      int sum =
          kernel.at<char>(0, 0) * img.at<cv::Vec3b>(r - 1, c - 1)[channel] +
          kernel.at<char>(0, 1) * img.at<cv::Vec3b>(r - 1, c)[channel] +
          kernel.at<char>(0, 2) * img.at<cv::Vec3b>(r - 1, c + 1)[channel] +
          kernel.at<char>(1, 0) * img.at<cv::Vec3b>(r, c - 1)[channel] +
          kernel.at<char>(1, 1) * img.at<cv::Vec3b>(r, c)[channel] +
          kernel.at<char>(1, 2) * img.at<cv::Vec3b>(r, c + 1)[channel] +
          kernel.at<char>(2, 0) * img.at<cv::Vec3b>(r + 1, c - 1)[channel] +
          kernel.at<char>(2, 1) * img.at<cv::Vec3b>(r + 1, c)[channel] +
          kernel.at<char>(2, 2) * img.at<cv::Vec3b>(r + 1, c + 1)[channel];

      pixel_weights.at<double>(r, c) = sum;
    }
  }

  return pixel_weights;
}


cv::Mat fs::Laplacian(const cv::Mat& img, int channel) {
  cv::Mat kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

  cv::Mat weights = ComputeWeights(img, kernel, channel);
  for (int r = 0; r < img.rows; ++r) {
    for (int c = 0; c < img.cols; ++c) {
      weights.at<double>(r, c) = std::abs(weights.at<double>(r, c));
    }
  }

  return weights;
}

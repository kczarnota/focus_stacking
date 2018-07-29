#include "focus_stacking.h"


cv::Mat fs::FocusStacking::ComputeSharpImage() {
  std::vector<cv::Mat> weights = ProcessImages(*images_);

  const int cols = (*images_)[0].cols;
  const int rows = (*images_)[0].rows;
  const size_t number_of_images = images_->size();
  cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC3);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      size_t max_ind = 0;
      double max_val = weights[0].at<double>(r, c);
      for (size_t i = 1; i < number_of_images; ++i) {
        if (weights[i].at<double>(r, c) > max_val) {
          max_val = weights[i].at<double>(r, c);
          max_ind = i;
        }
      }

      result.at<cv::Vec3b>(r, c) = (*images_)[max_ind].at<cv::Vec3b>(r, c);
    }
  }

  return result;
}


void fs::FocusStacking::GaussianBlur(cv::Mat* img) {
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


cv::Mat fs::FocusStacking::ComputeWeights(const cv::Mat& img,
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


cv::Mat fs::FocusStacking::Laplacian(const cv::Mat& img, int channel) {
  cv::Mat kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

  cv::Mat weights = ComputeWeights(img, kernel, channel);
  for (int r = 0; r < img.rows; ++r) {
    for (int c = 0; c < img.cols; ++c) {
      weights.at<double>(r, c) = std::abs(weights.at<double>(r, c));
    }
  }

  return weights;
}


std::vector<cv::Mat> fs::FocusStacking::ProcessImages(const std::vector<cv::Mat>& images) {
  const size_t number_of_images = images.size();
  std::vector<cv::Mat> processed_images(number_of_images);
  for (size_t i = 0; i < number_of_images; ++i) {
    processed_images[i] = images[i].clone();
  }

  std::vector<cv::Mat> weights(number_of_images);
  for (size_t i = 0; i < number_of_images; ++i) {
    GaussianBlur(&processed_images[i]);
      weights[i] = Laplacian(processed_images[i], static_cast<int>(selected_channel_));
  }

  return weights;
}

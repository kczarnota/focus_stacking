#include "focus_stacking.h"
#include <sstream>


const cv::Mat fs::FocusStacking::kGaussianKernel_ = (cv::Mat_<uchar>(3, 3) << 1, 2, 1,
                                                                              2, 4, 2,
                                                                              1, 2, 1);

const cv::Mat fs::FocusStacking::kLaplaceKernel_ = (cv::Mat_<char>(3, 3) << -1, -1, -1,
                                                                            -1,  8, -1,
                                                                            -1, -1, -1);

const cv::Mat fs::FocusStacking::kSobelXKernel_ = (cv::Mat_<char>(3, 3) << 1, 0, -1,
                                                                           2, 0, -2,
                                                                           1, 0, -1);

const cv::Mat fs::FocusStacking::kSobelYKernel_ = (cv::Mat_<char>(3, 3) << 1,  2,  1,
                                                                           0,  0,  0,
                                                                          -1, -2, -1);


std::ostream& operator<<(std::ostream& out, const fs::SelectedChannel value) {
  switch(value) {
    case fs::SelectedChannel::RED:
      return out << "RED";
    case fs::SelectedChannel::GREEN:
      return out << "GREEN";
    case fs::SelectedChannel::BLUE:
      return out << "BLUE";
    default:
      return out << "NO_SUCH_CHANNEL";
  }
}

std::ostream& operator<<(std::ostream& out, const fs::EdgeDetectionMethod value) {
  switch(value) {
    case fs::EdgeDetectionMethod::SOBEL:
      return out << "Sobel";
    case fs::EdgeDetectionMethod::LAPLACIAN:
      return out << "LAPLACIAN";
    default:
      return out << "NO_SUCH_METHOD";
  }
}

std::pair<cv::Mat, cv::Mat> fs::FocusStacking::ComputeSharpImageAndDepthMap() {
  std::vector<cv::Mat> weights = ProcessImages(*images_);

  const int cols = (*images_)[0].cols;
  const int rows = (*images_)[0].rows;
  const size_t number_of_images = images_->size();
  cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC3);
  cv::Mat depth = cv::Mat::zeros(rows, cols, CV_8UC1);

  std::vector<uchar> gray_colors =
      PrepareLookupTableWithColors(number_of_images, not_defined_depth_margin_);

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

      if (max_val < edge_threshold_) {
        max_ind = number_of_images - 1;
        depth.at<uchar>(r, c) = gray_colors[max_ind + 1];
      } else {
        depth.at<uchar>(r, c) = gray_colors[max_ind];
      }
      result.at<cv::Vec3b>(r, c) = (*images_)[max_ind].at<cv::Vec3b>(r, c);
    }
  }

  return std::make_pair(result, depth);
}


void fs::FocusStacking::PerformTests(std::vector<SelectedChannel> channels,
                  std::vector<EdgeDetectionMethod> methods,
                  std::vector<double> edge_thresholds,
                  std::vector<uchar> depth_margins) {
  for (SelectedChannel c : channels) {
    for (EdgeDetectionMethod m : methods) {
      for (double t : edge_thresholds) {
        for (uchar d : depth_margins) {
          fs::FocusStacking::SetChannel(c);
          fs::FocusStacking::SetEdgeDetectionMethod(m);
          fs::FocusStacking::SetEdgeThreshold(t);
          fs::FocusStacking::SetEdgeMargin(d);

          std::pair<cv::Mat, cv::Mat> result = ComputeSharpImageAndDepthMap();
          std::stringstream file_name_res;
          std::stringstream file_name_depth;
          file_name_res << "result_" << "_" << c << "_" << m << "_" << t << "_" << +d << ".png";
          file_name_depth << "depth_" << "_" << c << "_" << m << "_" << t << "_" << +d << ".png";
          cv::imwrite(file_name_res.str(), result.first);
          cv::imwrite(file_name_depth.str(), result.second);
        }
      }
    }

  }
}


void fs::FocusStacking::GaussianBlur(cv::Mat* img) {
  int sum_of_weights = 0;

  for (int r = 0; r < kGaussianKernel_.rows; ++r) {
    for (int c = 0; c < kGaussianKernel_.cols; ++c) {
      sum_of_weights += kGaussianKernel_.at<uchar>(r, c);
    }
  }

  cv::Mat img_copy = img->clone();
  for (int r = 1; r < img->rows - 1; ++r) {
    for (int c = 1; c < img->cols - 1; ++c) {
      for (int d = 0; d < 3; ++d) {
        int sum =
            kGaussianKernel_.at<uchar>(0, 0) * img_copy.at<cv::Vec3b>(r - 1, c - 1)[d] +
            kGaussianKernel_.at<uchar>(0, 1) * img_copy.at<cv::Vec3b>(r - 1, c)[d] +
            kGaussianKernel_.at<uchar>(0, 2) * img_copy.at<cv::Vec3b>(r - 1, c + 1)[d] +
            kGaussianKernel_.at<uchar>(1, 0) * img_copy.at<cv::Vec3b>(r, c - 1)[d] +
            kGaussianKernel_.at<uchar>(1, 1) * img_copy.at<cv::Vec3b>(r, c)[d] +
            kGaussianKernel_.at<uchar>(1, 2) * img_copy.at<cv::Vec3b>(r, c + 1)[d] +
            kGaussianKernel_.at<uchar>(2, 0) * img_copy.at<cv::Vec3b>(r + 1, c - 1)[d] +
            kGaussianKernel_.at<uchar>(2, 1) * img_copy.at<cv::Vec3b>(r + 1, c)[d] +
            kGaussianKernel_.at<uchar>(2, 2) * img_copy.at<cv::Vec3b>(r + 1, c + 1)[d];

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
  cv::Mat weights = ComputeWeights(img, kLaplaceKernel_, channel);
  for (int r = 0; r < img.rows; ++r) {
    for (int c = 0; c < img.cols; ++c) {
      weights.at<double>(r, c) = std::abs(weights.at<double>(r, c));
    }
  }

  return weights;
}


cv::Mat fs::FocusStacking::Sobel(const cv::Mat& img, int channel) {
  cv::Mat weights_x = ComputeWeights(img, kSobelXKernel_, channel);
  cv::Mat weights_y = ComputeWeights(img, kSobelYKernel_, channel);

  cv::Mat weights = cv::Mat(img.rows, img.cols, CV_64FC1);
  for (int r = 0; r < weights.rows; ++r) {
    for (int c = 0; c < weights.cols; ++c) {
      weights.at<double>(r, c) = sqrt(pow(weights_x.at<double>(r, c), 2) +
                                      pow(weights_y.at<double>(r, c), 2));
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
    if (edge_detection_method_ == EdgeDetectionMethod::LAPLACIAN)
      weights[i] =
          Laplacian(processed_images[i], static_cast<int>(selected_channel_));
    else
      weights[i] =
          Sobel(processed_images[i], static_cast<int>(selected_channel_));
  }

  return weights;
}


std::vector<uchar> fs::FocusStacking::PrepareLookupTableWithColors(
    size_t number_of_images, uchar margin) {
  std::vector<uchar> gray_colors(number_of_images + 1);
  size_t step = (kDephtColorMaxWalue_ - margin) / gray_colors.size();
  uchar color = kDephtColorMaxWalue_;
  for (size_t i = 0; i < gray_colors.size(); ++i) {
    gray_colors[i] = color;
    color -= step;
  }
  gray_colors[gray_colors.size() - 1] = 0;

  return gray_colors;
}

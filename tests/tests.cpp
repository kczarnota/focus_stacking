#include "image_loader.h"
#include "focus_stacking.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

const std::string kTestImagesPath = "../../tests/test_images";

cv::Mat CreateTestMatrix() {
  cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC3);
  img.at<cv::Vec3b>(0, 0)[0] = 1;
  img.at<cv::Vec3b>(0, 1)[0] = 2;
  img.at<cv::Vec3b>(0, 2)[0] = 1;
  img.at<cv::Vec3b>(1, 0)[0] = 2;
  img.at<cv::Vec3b>(1, 1)[0] = 4;
  img.at<cv::Vec3b>(1, 2)[0] = 2;
  img.at<cv::Vec3b>(2, 0)[0] = 1;
  img.at<cv::Vec3b>(2, 1)[0] = 2;
  img.at<cv::Vec3b>(2, 2)[0] = 1;
  return img;
}


TEST(test_image_loader, GetImagesPaths) {
  std::vector<std::string> paths = fs::GetImagesPaths(kTestImagesPath);
  EXPECT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0], "../../tests/test_images/b_bigbug0000_croppped.png");
}

TEST(test_image_loader, LoadImages) {
  auto images = fs::LoadImages(kTestImagesPath);
  EXPECT_EQ(images->size(), 1);
  EXPECT_EQ((*images)[0].rows, 228);
  EXPECT_EQ((*images)[0].cols, 300);
}

TEST(test_focus_stacking, GaussianBlur) {
  cv::Mat img = CreateTestMatrix();
  fs::FocusStacking::GaussianBlur(&img);
  EXPECT_EQ(img.at<cv::Vec3b>(0, 0)[0], 1);
  EXPECT_EQ(img.at<cv::Vec3b>(1, 1)[0], 2);
  EXPECT_EQ(img.at<cv::Vec3b>(1, 1)[1], 0);
}

TEST(test_focus_stacking, ComputeWeights) {
  cv::Mat img = CreateTestMatrix();
  cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 1, 2, 1,
                                             2, 4, 2,
                                             1, 2, 1);
  cv::Mat weights = fs::FocusStacking::ComputeWeights(img, kernel, 0);
  EXPECT_EQ(weights.at<double>(0, 0), 0);
  EXPECT_EQ(weights.at<double>(1, 1), 36);
}

TEST(test_focus_stacking, Laplacian) {
  cv::Mat img = CreateTestMatrix();
  cv::Mat weights = fs::FocusStacking::Laplacian(img, 0);
  EXPECT_EQ(weights.at<double>(0, 0), 0);
  EXPECT_EQ(weights.at<double>(1, 1), 20);
}

TEST(test_focus_stacking, PrepareLookupTableWithColors) {
  int number_of_images = 13;
  std::vector<uchar> gray_colors =
      fs::FocusStacking::PrepareLookupTableWithColors(number_of_images, 0);
  EXPECT_EQ(gray_colors[0], 255);
  EXPECT_EQ(gray_colors[1], 237);
  EXPECT_EQ(gray_colors[number_of_images - 1], 39);
  EXPECT_EQ(gray_colors[number_of_images], 0);
}

TEST(test_focus_stacking, Sobel) {
  cv::Mat img = CreateTestMatrix();
  cv::Mat weights = fs::FocusStacking::Sobel(img, 0);
  EXPECT_EQ(weights.at<double>(0, 0), 0);
  EXPECT_EQ(weights.at<double>(1, 1), 0);
}

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
  fs::GaussianBlur(&img);
  EXPECT_EQ(img.at<cv::Vec3b>(0, 0)[0], 1);
  EXPECT_EQ(img.at<cv::Vec3b>(1, 1)[0], 2);
  EXPECT_EQ(img.at<cv::Vec3b>(1, 1)[1], 0);
}

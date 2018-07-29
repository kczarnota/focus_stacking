#include "image_loader.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

const std::string kTestImagesPath = "../../tests/test_images";

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

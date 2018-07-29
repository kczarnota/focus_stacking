#include "image_loader.h"
#include <boost/filesystem.hpp>
#include <algorithm>

std::vector<std::string> fs::GetImagesPaths(const std::string& directory_path) {
  std::vector<std::string> paths;
  boost::filesystem::path root(directory_path);
  boost::filesystem::recursive_directory_iterator end;

  for (boost::filesystem::recursive_directory_iterator i(root); i != end; ++i) {
    const boost::filesystem::path cp = (*i);
    paths.push_back(cp.string());
  }

  std::sort(paths.begin(), paths.end());
  // The last image is result, which should not be used during stacking
  paths.erase(paths.end() - 1);

  return paths;
}

std::unique_ptr<std::vector<cv::Mat>> fs::LoadImages(
    const std::string& directory_path) {
  std::vector<std::string> paths = GetImagesPaths(directory_path);
  auto images = std::unique_ptr<std::vector<cv::Mat>>(new std::vector<cv::Mat>(paths.size()));

  for (size_t i = 0; i < paths.size(); ++i) {
    (*images)[i] = cv::imread(paths[i]);
  }

  return images;
}

#include "rkapp/capture/FolderSource.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace rkapp::capture {

FolderSource::FolderSource() = default;
FolderSource::~FolderSource() = default;

bool FolderSource::open(const std::string& folder_path) {
  namespace fs = std::filesystem;

  if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
    std::cerr << "Folder does not exist: " << folder_path << std::endl;
    return false;
  }

  folder_path_ = folder_path;
  image_files_.clear();
  current_index_ = 0;

  std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};

  for (const auto& entry : fs::directory_iterator(folder_path_)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

      if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
        image_files_.push_back(entry.path().string());
      }
    }
  }

  std::sort(image_files_.begin(), image_files_.end());
  is_opened_ = !image_files_.empty();

  if (image_files_.empty()) {
    std::cerr << "No image files found in folder: " << folder_path << std::endl;
  } else {
    std::cout << "Found " << image_files_.size() << " images in folder" << std::endl;
  }

  return is_opened_;
}

bool FolderSource::read(cv::Mat& frame) {
  if (!is_opened_ || current_index_ >= image_files_.size()) {
    return false;
  }

  std::string current_file = image_files_[current_index_];
  frame = cv::imread(current_file, cv::IMREAD_COLOR);

  if (frame.empty()) {
    std::cerr << "Failed to read image: " << current_file << std::endl;
    current_index_++;
    return false;
  }

  current_index_++;
  return true;
}

void FolderSource::release() {
  image_files_.clear();
  current_index_ = 0;
  is_opened_ = false;
}

bool FolderSource::isOpened() const { return is_opened_; }

double FolderSource::getFPS() const { return 30.0; }

cv::Size FolderSource::getSize() const {
  if (image_files_.empty()) return cv::Size(0, 0);
  cv::Mat sample = cv::imread(image_files_[0], cv::IMREAD_COLOR);
  return sample.empty() ? cv::Size(0, 0) : sample.size();
}

int FolderSource::getTotalFrames() const { return static_cast<int>(image_files_.size()); }

int FolderSource::getCurrentFrame() const { return static_cast<int>(current_index_); }

SourceType FolderSource::getType() const { return SourceType::FOLDER; }

} // namespace rkapp::capture

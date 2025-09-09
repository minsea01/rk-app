#pragma once

#include <opencv2/opencv.hpp>

namespace rkapp::preprocess {

struct LetterboxInfo {
    float scale;
    float dx, dy;
    int new_width, new_height;
};

class Preprocess {
public:
    static cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info);
    static cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info);
    
    static cv::Mat convertColor(const cv::Mat& src, int code = cv::COLOR_BGR2RGB);
    static cv::Mat normalize(const cv::Mat& src, float scale = 1.0f/255.0f);
    
    static cv::Mat hwc2chw(const cv::Mat& src);
    static cv::Mat blob(const cv::Mat& src);
};

} // namespace rkapp::preprocess
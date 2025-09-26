#include "rkapp/preprocess/Preprocess.hpp"
#include <algorithm>
#include <cstring>

namespace rkapp::preprocess {

cv::Mat Preprocess::letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info) {
    return letterbox(src, cv::Size(target_size, target_size), info);
}

cv::Mat Preprocess::letterbox(const cv::Mat& src, cv::Size target_size, LetterboxInfo& info) {
    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = target_size.width;
    int dst_h = target_size.height;
    
    // Calculate scale factor
    float scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    
    // New dimensions after scaling
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    
    // Calculate padding
    float dx = (dst_w - new_w) / 2.0f;
    float dy = (dst_h - new_h) / 2.0f;
    
    // Fill info struct
    info.scale = scale;
    info.dx = dx;
    info.dy = dy;
    info.new_width = new_w;
    info.new_height = new_h;
    
    // Resize image
    cv::Mat resized;
    if (scale != 1.0f) {
        cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = src.clone();
    }
    
    // Create output image with padding
    cv::Mat dst = cv::Mat::zeros(dst_h, dst_w, src.type());
    
    // Calculate integer offsets for padding
    int top = static_cast<int>(dy);
    int left = static_cast<int>(dx);
    
    // Copy resized image to center of padded image
    cv::Rect roi(left, top, new_w, new_h);
    resized.copyTo(dst(roi));
    
    return dst;
}

cv::Mat Preprocess::convertColor(const cv::Mat& src, int code) {
    cv::Mat dst;
    cv::cvtColor(src, dst, code);
    return dst;
}

cv::Mat Preprocess::normalize(const cv::Mat& src, float scale) {
    cv::Mat dst;
    src.convertTo(dst, CV_32F, scale);
    return dst;
}

cv::Mat Preprocess::hwc2chw(const cv::Mat& src) {
    if (src.channels() == 1) {
        return src.clone();
    }
    
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    
    // Create output matrix with CHW layout
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    
    cv::Mat dst(1, c * h * w, CV_32F);
    
    for (int i = 0; i < c; ++i) {
        std::memcpy(dst.ptr<float>() + i * h * w, 
                   channels[i].ptr<float>(), 
                   h * w * sizeof(float));
    }
    
    return dst;
}

cv::Mat Preprocess::blob(const cv::Mat& src) {
    // Convert BGR to RGB, normalize, and convert HWC to CHW
    cv::Mat rgb = convertColor(src, cv::COLOR_BGR2RGB);
    cv::Mat norm = normalize(rgb);
    return hwc2chw(norm);
}

} // namespace rkapp::preprocess
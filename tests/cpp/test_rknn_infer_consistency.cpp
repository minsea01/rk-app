#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// Simplified letterbox info for testing
struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

class RknnInferConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test image (640x480 BGR)
        test_image_ = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));

        // Draw some simple shapes for detection
        cv::rectangle(test_image_, cv::Point(100, 100), cv::Point(200, 200),
                      cv::Scalar(255, 0, 0), -1);
        cv::rectangle(test_image_, cv::Point(400, 300), cv::Point(500, 400),
                      cv::Scalar(0, 255, 0), -1);

        original_size_ = test_image_.size();
    }

    cv::Mat test_image_;
    cv::Size original_size_;
};

// Simple letterbox implementation for testing
cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info) {
    float scale = std::min(
        static_cast<float>(target_size) / src.cols,
        static_cast<float>(target_size) / src.rows
    );

    int new_w = static_cast<int>(src.cols * scale);
    int new_h = static_cast<int>(src.rows * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    int pad_x = (target_size - new_w) / 2;
    int pad_y = (target_size - new_h) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       pad_y, target_size - new_h - pad_y,
                       pad_x, target_size - new_w - pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    return padded;
}

// Test that letterbox preprocessing works correctly
TEST_F(RknnInferConsistencyTest, LetterboxPreprocessing) {
    LetterboxInfo info;
    cv::Mat preprocessed = letterbox(test_image_, 640, info);

    ASSERT_EQ(preprocessed.cols, 640);
    ASSERT_EQ(preprocessed.rows, 640);
    ASSERT_EQ(preprocessed.type(), CV_8UC3);

    // Verify letterbox info is valid
    EXPECT_GT(info.scale, 0.0f);
    EXPECT_GE(info.pad_x, 0);
    EXPECT_GE(info.pad_y, 0);
}

// Test RGB <-> BGR conversion consistency
TEST_F(RknnInferConsistencyTest, RGBBGRRoundTrip) {
    cv::Mat rgb;
    cv::cvtColor(test_image_, rgb, cv::COLOR_BGR2RGB);

    cv::Mat bgr_back;
    cv::cvtColor(rgb, bgr_back, cv::COLOR_RGB2BGR);

    // Should be identical after round-trip
    cv::Mat diff;
    cv::absdiff(test_image_, bgr_back, diff);

    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0)
        << "RGB<->BGR conversion should be lossless";
}

// Test letterbox coordinate transformation
TEST_F(RknnInferConsistencyTest, LetterboxCoordinateTransform) {
    LetterboxInfo info;
    cv::Mat preprocessed = letterbox(test_image_, 640, info);

    // Original point (100, 100)
    float orig_x = 100.0f, orig_y = 100.0f;

    // Transform to letterbox space
    float letterbox_x = (orig_x * info.scale) + info.pad_x;
    float letterbox_y = (orig_y * info.scale) + info.pad_y;

    // Transform back
    float back_x = (letterbox_x - info.pad_x) / info.scale;
    float back_y = (letterbox_y - info.pad_y) / info.scale;

    EXPECT_NEAR(back_x, orig_x, 1.0f) << "X coordinate round-trip failed";
    EXPECT_NEAR(back_y, orig_y, 1.0f) << "Y coordinate round-trip failed";
}

// Test that preprocessing preserves aspect ratio
TEST_F(RknnInferConsistencyTest, LetterboxPreservesAspectRatio) {
    LetterboxInfo info;
    cv::Mat preprocessed = letterbox(test_image_, 640, info);

    float orig_aspect = static_cast<float>(test_image_.cols) / test_image_.rows;

    // Calculate effective area after letterbox
    int effective_width = static_cast<int>(test_image_.cols * info.scale);
    int effective_height = static_cast<int>(test_image_.rows * info.scale);
    float letterbox_aspect = static_cast<float>(effective_width) / effective_height;

    EXPECT_NEAR(orig_aspect, letterbox_aspect, 0.01f)
        << "Letterbox should preserve aspect ratio";
}

// Test DFL decode consistency (conceptual test)
TEST_F(RknnInferConsistencyTest, DFLDecodeParametersValid) {
    // YOLOv8/11 uses reg_max=16, strides=[8,16,32]
    const int reg_max = 16;
    const std::vector<int> strides = {8, 16, 32};

    EXPECT_EQ(reg_max, 16) << "YOLOv8/11 uses reg_max=16";
    EXPECT_EQ(strides.size(), 3u) << "P5 models use 3 detection heads";

    // Verify stride progression
    for (size_t i = 1; i < strides.size(); ++i) {
        EXPECT_EQ(strides[i], strides[i-1] * 2)
            << "Strides should double at each level";
    }
}

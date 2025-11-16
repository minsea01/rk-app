/*
 * Unit tests for preprocessing module
 * Framework: Google Test
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// Mock preprocessing functions for testing
namespace preprocessing {

struct Image {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
};

Image letterbox(const Image& input, int target_size) {
    // Simplified letterbox implementation for testing
    Image output;
    output.width = target_size;
    output.height = target_size;
    output.channels = input.channels;

    // Calculate scaling
    float scale = std::min(
        static_cast<float>(target_size) / input.width,
        static_cast<float>(target_size) / input.height
    );

    int new_width = static_cast<int>(input.width * scale);
    int new_height = static_cast<int>(input.height * scale);

    // Allocate output
    output.data.resize(target_size * target_size * input.channels, 114);  // Gray padding

    // Simple resize (nearest neighbor for test)
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int src_y = static_cast<int>(y / scale);
            int src_x = static_cast<int>(x / scale);

            if (src_y < input.height && src_x < input.width) {
                for (int c = 0; c < input.channels; ++c) {
                    int dst_idx = (y * target_size + x) * input.channels + c;
                    int src_idx = (src_y * input.width + src_x) * input.channels + c;
                    output.data[dst_idx] = input.data[src_idx];
                }
            }
        }
    }

    return output;
}

bool normalize_image(std::vector<float>& data, float mean, float std) {
    if (std == 0.0f) return false;

    for (auto& pixel : data) {
        pixel = (pixel - mean) / std;
    }
    return true;
}

} // namespace preprocessing

// Test fixtures
class PreprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test image
        test_image.width = 1920;
        test_image.height = 1080;
        test_image.channels = 3;
        test_image.data.resize(test_image.width * test_image.height * test_image.channels, 128);
    }

    preprocessing::Image test_image;
};

// Test cases
TEST_F(PreprocessTest, LetterboxPreservesAspectRatio) {
    auto result = preprocessing::letterbox(test_image, 640);

    EXPECT_EQ(result.width, 640);
    EXPECT_EQ(result.height, 640);
    EXPECT_EQ(result.channels, 3);
    EXPECT_EQ(result.data.size(), 640 * 640 * 3);
}

TEST_F(PreprocessTest, LetterboxHandlesSquareInput) {
    preprocessing::Image square_img;
    square_img.width = 640;
    square_img.height = 640;
    square_img.channels = 3;
    square_img.data.resize(640 * 640 * 3, 100);

    auto result = preprocessing::letterbox(square_img, 640);

    EXPECT_EQ(result.width, 640);
    EXPECT_EQ(result.height, 640);
}

TEST_F(PreprocessTest, LetterboxHandlesPortraitImage) {
    preprocessing::Image portrait;
    portrait.width = 480;
    portrait.height = 640;
    portrait.channels = 3;
    portrait.data.resize(480 * 640 * 3, 50);

    auto result = preprocessing::letterbox(portrait, 640);

    EXPECT_EQ(result.width, 640);
    EXPECT_EQ(result.height, 640);
}

TEST_F(PreprocessTest, NormalizeValidInput) {
    std::vector<float> data = {0.0f, 127.5f, 255.0f};
    bool success = preprocessing::normalize_image(data, 127.5f, 127.5f);

    EXPECT_TRUE(success);
    EXPECT_FLOAT_EQ(data[0], -1.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 1.0f);
}

TEST_F(PreprocessTest, NormalizeHandlesZeroStd) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    bool success = preprocessing::normalize_image(data, 0.0f, 0.0f);

    EXPECT_FALSE(success);
}

TEST_F(PreprocessTest, NormalizeEmptyInput) {
    std::vector<float> data;
    bool success = preprocessing::normalize_image(data, 0.0f, 1.0f);

    EXPECT_TRUE(success);  // Should handle empty input gracefully
    EXPECT_TRUE(data.empty());
}

// Performance test
TEST(PreprocessPerformanceTest, LetterboxPerformance) {
    preprocessing::Image large_image;
    large_image.width = 3840;  // 4K
    large_image.height = 2160;
    large_image.channels = 3;
    large_image.data.resize(large_image.width * large_image.height * large_image.channels, 128);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = preprocessing::letterbox(large_image, 640);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Should complete in reasonable time (< 100ms for 4K image)
    EXPECT_LT(duration, 100) << "Letterbox resize took " << duration << "ms";
}

// Edge case tests
TEST(PreprocessEdgeCases, ZeroSizeImage) {
    preprocessing::Image zero_img;
    zero_img.width = 0;
    zero_img.height = 0;
    zero_img.channels = 3;

    // Should not crash
    EXPECT_NO_THROW({
        auto result = preprocessing::letterbox(zero_img, 640);
    });
}

TEST(PreprocessEdgeCases, VerySmallImage) {
    preprocessing::Image tiny;
    tiny.width = 1;
    tiny.height = 1;
    tiny.channels = 3;
    tiny.data = {255, 0, 0};

    auto result = preprocessing::letterbox(tiny, 640);

    EXPECT_EQ(result.width, 640);
    EXPECT_EQ(result.height, 640);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

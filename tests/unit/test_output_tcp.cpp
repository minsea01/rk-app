#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include "rkapp/output/TcpOutput.hpp"

using rkapp::output::FrameResult;
using rkapp::output::TcpOutput;

namespace {

FrameResult makeResult(int frame_id) {
    FrameResult r{};
    r.frame_id = frame_id;
    r.timestamp = frame_id * 100;
    r.width = 640;
    r.height = 480;
    r.source_uri = "dummy";
    return r;
}

}  // namespace

TEST(TcpOutputTest, QueuesWhenServerUnavailable) {
    TcpOutput output;
    ASSERT_TRUE(output.open("127.0.0.1:65533,queue:2,backoff:50"));
    EXPECT_TRUE(output.isOpened());
    EXPECT_FALSE(output.isConnected());

    EXPECT_FALSE(output.send(makeResult(1)));
    EXPECT_EQ(output.backlogDepth(), 1U);

    EXPECT_FALSE(output.send(makeResult(2)));
    EXPECT_EQ(output.backlogDepth(), 2U);

    EXPECT_FALSE(output.send(makeResult(3)));
    EXPECT_EQ(output.backlogDepth(), 2U);

    output.close();
}

TEST(TcpOutputTest, BackoffGrowUntilMax) {
    TcpOutput output;
    ASSERT_TRUE(output.open("127.0.0.1:65530,backoff:50,backoff_max:150"));
    EXPECT_TRUE(output.isOpened());

    EXPECT_FALSE(output.send(makeResult(1)));
    EXPECT_EQ(output.reconnectBackoff().count(), 100);

    std::this_thread::sleep_for(output.reconnectBackoff() + std::chrono::milliseconds(20));
    EXPECT_FALSE(output.send(makeResult(2)));
    EXPECT_EQ(output.reconnectBackoff().count(), 150);

    std::this_thread::sleep_for(output.reconnectBackoff() + std::chrono::milliseconds(20));
    EXPECT_FALSE(output.send(makeResult(3)));
    EXPECT_EQ(output.reconnectBackoff().count(), 150);

    output.close();
}

TEST(TcpOutputTest, SerializesEmbeddedImagePayloadToJsonFile) {
    namespace fs = std::filesystem;

    const auto unique =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path out_path =
        fs::temp_directory_path() / ("rkapp_tcp_output_" + std::to_string(unique) + ".jsonl");

    TcpOutput output;
    ASSERT_TRUE(output.open("127.0.0.1:65533,file:" + out_path.string()));

    FrameResult result = makeResult(7);
    result.image_encoding = "jpeg";
    result.image_contains_overlays = true;
    result.image_bytes = {0x01, 0x02, 0x03};

    EXPECT_TRUE(output.send(result));
    output.close();

    std::ifstream handle(out_path);
    ASSERT_TRUE(handle.is_open());
    std::string line;
    std::getline(handle, line);
    EXPECT_NE(line.find("\"frame_id\":7"), std::string::npos);
    EXPECT_NE(line.find("\"image\""), std::string::npos);
    EXPECT_NE(line.find("\"encoding\":\"jpeg\""), std::string::npos);
    EXPECT_NE(line.find("\"contains_overlays\":true"), std::string::npos);
    EXPECT_NE(line.find("\"data_base64\":\"AQID\""), std::string::npos);

    fs::remove(out_path);
}

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "rkapp/infer/IInferEngine.hpp"
#include "rkapp/infer/ModelMeta.hpp"

#define private public
#include "rkapp/infer/RknnEngine.hpp"
#undef private

namespace fs = std::filesystem;

namespace {

fs::path makeTempDir(const std::string& prefix) {
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  fs::path dir = fs::temp_directory_path() / (prefix + std::to_string(unique));
  fs::create_directories(dir);
  return dir;
}

}  // namespace

TEST(RknnEngineLocalTest, InitUsesModelSpecMetadataOverConflictingSidecar) {
  const fs::path dir = makeTempDir("rkapp_rknn_engine_local_");
  const fs::path model = dir / "demo.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"dfl","reg_max":16,"strides":[8,16,32],"num_classes":80})";

  rkapp::infer::RknnEngine engine;
  rkapp::infer::ModelSpec spec;
  spec.backend = rkapp::infer::ModelBackend::RKNN;
  spec.model_path = model.string();
  spec.input_size = 640;
  spec.decode_meta.head = "raw";
  spec.decode_meta.num_classes = 1;
  spec.decode_meta.has_objectness = 0;
  spec.decode_meta.output_index = 0;
  spec.decode_meta_path = "model_spec://unit-test";

  ASSERT_TRUE(engine.init(spec));

  EXPECT_EQ(engine.model_meta_.head, "raw");
  EXPECT_EQ(engine.model_meta_.num_classes, 1);
  EXPECT_EQ(engine.model_meta_.has_objectness, 0);
  EXPECT_EQ(engine.model_meta_.output_index, 0);
  EXPECT_EQ(engine.model_meta_source_, "model_spec://unit-test");

  engine.release();
  fs::remove_all(dir);
}

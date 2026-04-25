#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "rkapp/infer/ModelMeta.hpp"
#include "rkapp/infer/OnnxEngine.hpp"
#include "rkapp/pipeline/DetectionPipeline.hpp"

namespace fs = std::filesystem;

namespace {

fs::path repoRoot() {
  return fs::path(__FILE__).parent_path().parent_path().parent_path();
}

fs::path makeTempDir(const std::string& prefix) {
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  fs::path dir = fs::temp_directory_path() / (prefix + std::to_string(unique));
  fs::create_directories(dir);
  return dir;
}

}  // namespace

TEST(PipelineConfigNormalize, ResolvesStructuredModelSpecAndSidecar) {
  const fs::path dir = makeTempDir("rkapp_cfg_normalize_");
  const fs::path model = dir / "demo_person.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"raw","num_classes":1,"has_objectness":0,"score_is_probability":1})";

  rkapp::pipeline::PipelineConfig config;
  config.source.uri = "assets";
  config.source.type = rkapp::capture::SourceType::FOLDER;
  config.source.use_mpp_decode = false;
  config.model.model_path = model.string();
  config.model.input_size = 640;
  config.model.conf_threshold = 0.42f;
  config.model.iou_threshold = 0.33f;
  config.model.max_detections = 77;
  config.model.min_box_size = 12.0f;
  config.model.max_box_size = 320.0f;
  config.model.min_aspect_ratio = 0.25f;
  config.model.max_aspect_ratio = 3.0f;
  config.model.use_npu_multicore = false;
  config.model.use_zero_copy = false;
  config.preprocess.use_rga_preprocess = false;
  config.preprocess.enable_undistort = true;
  config.preprocess.calibration_file = "camera.yaml";
  config.preprocess.profile = "quality";
  config.output.enable_profiling = true;

  std::vector<std::string> warnings;
  const auto normalized = rkapp::pipeline::normalizePipelineConfig(config, &warnings);

  EXPECT_TRUE(warnings.empty());
  EXPECT_EQ(normalized.source.uri, "assets");
  EXPECT_EQ(normalized.source.type, rkapp::capture::SourceType::FOLDER);
  EXPECT_FALSE(normalized.source.use_mpp_decode);
  EXPECT_EQ(normalized.model.model_path, model.string());
  EXPECT_EQ(normalized.model.input_size, 640);
  EXPECT_FLOAT_EQ(normalized.model.conf_threshold, 0.42f);
  EXPECT_FLOAT_EQ(normalized.model.iou_threshold, 0.33f);
  EXPECT_EQ(normalized.model.max_detections, 77);
  EXPECT_FLOAT_EQ(normalized.model.min_box_size, 12.0f);
  EXPECT_FLOAT_EQ(normalized.model.max_box_size, 320.0f);
  EXPECT_FLOAT_EQ(normalized.model.min_aspect_ratio, 0.25f);
  EXPECT_FLOAT_EQ(normalized.model.max_aspect_ratio, 3.0f);
  EXPECT_FALSE(normalized.model.use_npu_multicore);
  EXPECT_FALSE(normalized.model.use_zero_copy);
  EXPECT_FALSE(normalized.preprocess.use_rga_preprocess);
  EXPECT_TRUE(normalized.preprocess.enable_undistort);
  EXPECT_EQ(normalized.preprocess.calibration_file, "camera.yaml");
  EXPECT_EQ(normalized.preprocess.profile, "quality");
  EXPECT_TRUE(normalized.output.enable_profiling);
  EXPECT_EQ(normalized.model.decode_meta.num_classes, 1);
  EXPECT_EQ(normalized.model.decode_meta.has_objectness, 0);
  EXPECT_EQ(normalized.model.decode_meta.head, "raw");
  EXPECT_EQ(normalized.model.decode_meta.score_is_probability, 1);
  EXPECT_NE(normalized.model.decode_meta_path.find("demo_person.rknn.json"),
            std::string::npos);

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, LoadsCanonicalStructuredSchema) {
  const fs::path dir = makeTempDir("rkapp_cfg_loader_");
  const fs::path model = dir / "demo.onnx";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"dfl","reg_max":16,"strides":[8,16,32],"num_classes":3})";

  const fs::path config_path = dir / "detect.yaml";
  std::ofstream(config_path) << R"(
source:
  type: video
  uri: "clip.mp4"
  use_mpp_decode: false
engine:
  type: onnx
  model: "demo.onnx"
  input_size: [416, 416]
postprocess:
  conf_threshold: 0.4
  nms_threshold: 0.3
  max_detections: 25
  min_box_size: 16
  aspect_ratio_range: [0.4, 2.5]
classes:
  names: [person, bicycle, car]
preprocess:
  profile: balanced
tracking:
  enable: true
  match_iou: 0.35
  ema_alpha: 0.7
  confirm_hits: 3
  max_misses: 5
  keep_missing_tracks: false
  missing_conf_decay: 0.12
output:
  type: tcp
  enable_profiling: true
  tcp:
    host: "127.0.0.1"
    port: 9010
    queue_size: 8
    bind_ip: "192.168.20.10"
    iface: "eth1"
    include_image: true
    image_quality: 72
    image_interval: 3
    draw_detections: false
runtime:
  warmup: 7
  async: true
logging:
  level: "DEBUG"
failure:
  frame_error_limit: 4
  source_recovery_grace_ms: 2500
  max_reconnect_attempts: 6
  initial_backoff_ms: 150
  max_backoff_ms: 1200
)";

  const auto loaded = rkapp::pipeline::loadPipelineConfigFile(config_path.string());

  EXPECT_TRUE(loaded.warnings.empty());
  EXPECT_EQ(loaded.config.source.type, rkapp::capture::SourceType::VIDEO);
  EXPECT_EQ(loaded.config.source.uri, (dir / "clip.mp4").string());
  EXPECT_FALSE(loaded.config.source.use_mpp_decode);
  EXPECT_EQ(loaded.config.model.backend, rkapp::pipeline::PipelineConfig::ModelBackend::ONNX);
  EXPECT_EQ(loaded.config.model.input_size, 416);
  EXPECT_FLOAT_EQ(loaded.config.model.conf_threshold, 0.4f);
  EXPECT_FLOAT_EQ(loaded.config.model.iou_threshold, 0.3f);
  EXPECT_EQ(loaded.config.model.max_detections, 25);
  EXPECT_FLOAT_EQ(loaded.config.model.min_box_size, 16.0f);
  EXPECT_FLOAT_EQ(loaded.config.model.min_aspect_ratio, 0.4f);
  EXPECT_FLOAT_EQ(loaded.config.model.max_aspect_ratio, 2.5f);
  EXPECT_EQ(loaded.config.model.class_names.size(), 3u);
  EXPECT_EQ(loaded.config.preprocess.profile, "balanced");
  EXPECT_TRUE(loaded.config.tracking.enable);
  EXPECT_FLOAT_EQ(loaded.config.tracking.match_iou, 0.35f);
  EXPECT_FLOAT_EQ(loaded.config.tracking.ema_alpha, 0.7f);
  EXPECT_EQ(loaded.config.tracking.confirm_hits, 3);
  EXPECT_EQ(loaded.config.tracking.max_misses, 5);
  EXPECT_FALSE(loaded.config.tracking.keep_missing_tracks);
  EXPECT_FLOAT_EQ(loaded.config.tracking.missing_conf_decay, 0.12f);
  EXPECT_TRUE(loaded.config.output.enable_profiling);
  EXPECT_EQ(loaded.config.output.host, "127.0.0.1");
  EXPECT_EQ(loaded.config.output.port, 9010);
  EXPECT_EQ(loaded.config.output.queue_size, 8);
  EXPECT_EQ(loaded.config.output.bind_ip, "192.168.20.10");
  EXPECT_EQ(loaded.config.output.bind_interface, "eth1");
  EXPECT_TRUE(loaded.config.output.include_image);
  EXPECT_EQ(loaded.config.output.image_quality, 72);
  EXPECT_EQ(loaded.config.output.image_interval, 3);
  EXPECT_FALSE(loaded.config.output.draw_detections);
  EXPECT_EQ(loaded.config.runtime.warmup_iterations, 7);
  EXPECT_TRUE(loaded.config.runtime.async_mode);
  EXPECT_EQ(loaded.config.logging.level, "DEBUG");
  EXPECT_EQ(loaded.config.failure.frame_error_limit, 4);
  EXPECT_EQ(loaded.config.failure.source_recovery_grace_ms, 2500);
  EXPECT_EQ(loaded.config.failure.max_reconnect_attempts, 6);
  EXPECT_EQ(loaded.config.failure.initial_backoff_ms, 150);
  EXPECT_EQ(loaded.config.failure.max_backoff_ms, 1200);
  EXPECT_EQ(loaded.config.model.decode_meta.head, "dfl");
  EXPECT_EQ(loaded.config.model.decode_meta.reg_max, 16);
  EXPECT_EQ(loaded.config.model.decode_meta.num_classes, 3);
  EXPECT_NE(loaded.config.model.decode_meta_path.find("demo.onnx.json"), std::string::npos);

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, KeepsCsiUriStructuredStringUnresolved) {
  const fs::path dir = makeTempDir("rkapp_cfg_loader_csi_");
  const fs::path model = dir / "demo.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"raw","num_classes":1,"has_objectness":0,"output_index":0})";

  const fs::path config_path = dir / "detect.yaml";
  std::ofstream(config_path) << R"(
source:
  type: csi
  uri: "device=/dev/video0,width=1920,height=1080,framerate=30,format=NV12"
engine:
  type: rknn
  model: "demo.rknn"
  input_size: [640, 640]
runtime:
  warmup: 10
  async: true
)";

  const auto loaded = rkapp::pipeline::loadPipelineConfigFile(config_path.string());

  EXPECT_TRUE(loaded.warnings.empty());
  EXPECT_EQ(loaded.config.source.type, rkapp::capture::SourceType::CSI);
  EXPECT_EQ(loaded.config.source.uri,
            "device=/dev/video0,width=1920,height=1080,framerate=30,format=NV12");
  EXPECT_EQ(loaded.config.runtime.warmup_iterations, 10);
  EXPECT_TRUE(loaded.config.runtime.async_mode);
  EXPECT_EQ(loaded.config.model.model_path, model.string());

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, ResolvesRepoRootRelativePathsFromNestedConfigDir) {
  const fs::path dir = makeTempDir("rkapp_cfg_loader_repo_root_");
  std::ofstream(dir / "CMakeLists.txt") << "cmake_minimum_required(VERSION 3.16)\n";
  std::ofstream(dir / "CMakePresets.json") << "{}\n";

  const fs::path model_dir = dir / "artifacts/models";
  const fs::path assets_dir = dir / "assets";
  const fs::path classes_dir = dir / "config";
  const fs::path cfg_dir = dir / "config/detection";
  fs::create_directories(model_dir);
  fs::create_directories(assets_dir);
  fs::create_directories(classes_dir);
  fs::create_directories(cfg_dir);

  const fs::path model = model_dir / "demo.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"raw","num_classes":1,"has_objectness":0,"output_index":0})";
  std::ofstream(assets_dir / "sample.jpg").put('\n');
  std::ofstream(classes_dir / "person_classes.txt") << "person\n";

  const fs::path config_path = cfg_dir / "detect.yaml";
  std::ofstream(config_path) << R"(
source:
  type: folder
  uri: "assets"
engine:
  type: rknn
  model: "artifacts/models/demo.rknn"
  input_size: [640, 640]
classes: "config/person_classes.txt"
)";

  const auto loaded = rkapp::pipeline::loadPipelineConfigFile(config_path.string());

  EXPECT_TRUE(loaded.warnings.empty());
  EXPECT_EQ(loaded.config.source.uri, (dir / "assets").string());
  EXPECT_EQ(loaded.config.model.model_path, model.string());
  ASSERT_EQ(loaded.config.model.class_names.size(), 1u);
  EXPECT_EQ(loaded.config.model.class_names.front(), "person");
  EXPECT_EQ(loaded.config.model.decode_meta.head, "raw");

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, RejectsDeprecatedInputAndNmsAliases) {
  const fs::path dir = makeTempDir("rkapp_cfg_loader_deprecated_");
  const fs::path config_path = dir / "detect.yaml";
  std::ofstream(config_path) << R"(
input:
  type: video
  video:
    path: "clip.mp4"
engine:
  type: onnx
  imgsz: 416
  model: "demo.onnx"
nms:
  conf_thres: 0.4
)";

  EXPECT_THROW(
      {
        try {
          (void)rkapp::pipeline::loadPipelineConfigFile(config_path.string());
        } catch (const std::runtime_error& e) {
          EXPECT_NE(std::string(e.what()).find("Deprecated config key"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, ModelLocalSidecarOverridesMissingMetadata) {
  const fs::path dir = makeTempDir("rkapp_cfg_loader_override_");
  const fs::path model = dir / "demo.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"raw","num_classes":1,"has_objectness":0,"output_index":0})";
  std::ofstream(model.string() + ".meta")
      << "task=detect\nnum_classes=1\nhas_objectness=false\n";

  rkapp::pipeline::PipelineConfig config;
  config.model.model_path = model.string();
  const auto normalized = rkapp::pipeline::normalizePipelineConfig(config);

  EXPECT_EQ(normalized.model.decode_meta.head, "raw");
  EXPECT_EQ(normalized.model.decode_meta.num_classes, 1);
  EXPECT_EQ(normalized.model.decode_meta.has_objectness, 0);
  EXPECT_EQ(normalized.model.decode_meta.output_index, 0);
  EXPECT_NE(normalized.model.decode_meta_path.find("demo.rknn.json"), std::string::npos);

  fs::remove_all(dir);
}

TEST(PipelineConfigLoader, LoadsModelLocalSidecarForRknnModel) {
  const fs::path dir = makeTempDir("rkapp_model_meta_");
  const fs::path model = dir / "demo_person.rknn";
  std::ofstream(model).put('\n');
  std::ofstream(model.string() + ".json")
      << R"({"head":"raw","num_classes":1,"has_objectness":0,"score_is_probability":1,)"
         R"("output_index":0})";

  const auto loaded = rkapp::infer::loadModelMetaFromPath(model.string());
  EXPECT_EQ(loaded.meta.head, "raw");
  EXPECT_EQ(loaded.meta.num_classes, 1);
  EXPECT_EQ(loaded.meta.has_objectness, 0);
  EXPECT_EQ(loaded.meta.score_is_probability, 1);
  EXPECT_EQ(loaded.meta.output_index, 0);
  EXPECT_NE(loaded.source_path.find("demo_person.rknn.json"), std::string::npos);

  fs::remove_all(dir);
}

TEST(PipelineFactory, CreateEngineRespectsExplicitBackend) {
  rkapp::pipeline::PipelineConfig::ModelSpec model;
  model.backend = rkapp::pipeline::PipelineConfig::ModelBackend::ONNX;
  model.model_path = "model.bin";
  model.decode_meta.head = "raw";
  model.decode_meta.num_classes = 1;
  model.decode_meta.has_objectness = 0;
  model.decode_meta.score_is_probability = 1;

  auto engine = rkapp::pipeline::createEngine(model);
#if RKAPP_WITH_ONNX
  ASSERT_NE(engine, nullptr);
  EXPECT_NE(dynamic_cast<rkapp::infer::OnnxEngine*>(engine.get()), nullptr);
#else
  EXPECT_EQ(engine, nullptr);
#endif
}

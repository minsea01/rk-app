#include <gtest/gtest.h>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/pipeline/DetectionPipeline.hpp"

namespace {

TEST(DetectionPipelineFactoryTest, CreateSourceHandlesGigeType) {
  rkapp::pipeline::PipelineConfig config;
  config.source.type = rkapp::capture::SourceType::GIGE;

  auto source = rkapp::pipeline::createSource(config);
#if RKAPP_WITH_GIGE
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::GIGE);
#else
  EXPECT_EQ(source, nullptr);
#endif
}

TEST(DetectionPipelineFactoryTest, CreateSourceHandlesCsiType) {
  rkapp::pipeline::PipelineConfig config;
  config.source.type = rkapp::capture::SourceType::CSI;

  auto source = rkapp::pipeline::createSource(config);
#if RKAPP_WITH_CSI
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::CSI);
#else
  EXPECT_EQ(source, nullptr);
#endif
}

}  // namespace

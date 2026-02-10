#include <gtest/gtest.h>

#include "rkapp/capture/ISource.hpp"
#include "rkapp/pipeline/DetectionPipeline.hpp"

namespace {

TEST(DetectionPipelineFactoryTest, CreateSourceHandlesGigeType) {
  rkapp::pipeline::PipelineConfig config;
  config.source_type = rkapp::capture::SourceType::GIGE;

  auto source = rkapp::pipeline::createSource(config);
  ASSERT_NE(source, nullptr);
#if RKAPP_WITH_GIGE
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::GIGE);
#else
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::VIDEO);
#endif
}

TEST(DetectionPipelineFactoryTest, CreateSourceHandlesCsiType) {
  rkapp::pipeline::PipelineConfig config;
  config.source_type = rkapp::capture::SourceType::CSI;

  auto source = rkapp::pipeline::createSource(config);
  ASSERT_NE(source, nullptr);
#if RKAPP_WITH_CSI
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::CSI);
#else
  EXPECT_EQ(source->getType(), rkapp::capture::SourceType::VIDEO);
#endif
}

}  // namespace

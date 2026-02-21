// RKNN 推理引擎头文件（面向初学者说明）
// 采用 PImpl 设计：把 RKNN SDK 细节放到 .cpp，减少头文件依赖和编译耦合。
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rkapp/infer/IInferEngine.hpp"

// 前置声明：零拷贝路径会使用 DMA-BUF 描述符，不在此处引入具体实现。
namespace rkapp::common {
class DmaBuf;
}

namespace rkapp::preprocess {
struct LetterboxInfo;
}

namespace rkapp::infer {

// 模型元信息（通常来自模型旁路 metadata 文件）
// 用于决定输出张量如何解码（raw/dfl）、类别数、输出索引等。
struct ModelMeta {
  int reg_max = -1;
  std::vector<int> strides;
  std::string head; // "dfl" / "raw" / ""（空表示未知）
  int output_index = -1;  // 多输出模型时，指定用于检测的输出分支
  int num_classes = -1;   // 类别数（建议显式填写，避免推断歧义）
  int has_objectness = -1;  // -1 未知，0 无 obj 分支，1 有 obj 分支
  std::string task{"detect"};  // "detect" | "pose"
  int num_keypoints = 0;       // 姿态模型关键点数量（纯检测时为 0）
};

class RknnEngine : public IInferEngine {
public:
  RknnEngine();
  ~RknnEngine() override;

  // 初始化模型与运行时上下文。
  // img_size 用于与模型输入形状进行一致性检查。
  bool init(const std::string& model_path, int img_size = 640) override;

  // 通用推理入口：内部会做 letterbox 预处理，再调用 inferPreprocessed。
  std::vector<Detection> infer(const cv::Mat& image) override;

  /**
   * @brief 对“已经完成预处理（letterbox）”的图像执行推理
   *
   * 当外部流程已经做过缩放补边时，调用该接口可避免重复预处理。
   *
   * @param preprocessed_image 已 letterbox 的图像（尺寸应为 input_size_ x input_size_）
   * @param original_size 预处理前原图尺寸
   * @param letterbox_info letterbox 参数（用于将框坐标映射回原图）
   * @return 检测结果（坐标系为原图坐标）
   */
  std::vector<Detection> inferPreprocessed(
      const cv::Mat& preprocessed_image,
      const cv::Size& original_size,
      const struct rkapp::preprocess::LetterboxInfo& letterbox_info);

  // 预热：用一张虚拟图跑一次，减少首次推理抖动。
  void warmup() override;
  // 释放 RKNN 资源；析构时也会调用。
  void release() override;

  /**
   * @brief 从 DMA-BUF 内存执行零拷贝推理
   *
   * 通过将 DMA-BUF fd 直接导入 RKNN，尽量减少 CPU 拷贝开销。
   * 该路径需要编译期启用 ENABLE_RKNN_DMA_FD。
   * 如果启用 ENABLE_RKNN_IO_MEM（RKNN SDK >= 1.5.0），可进一步优化 IO 绑定。
   *
   * 支持输入格式：
   * - RGB888（3 通道，且已 letterbox 到 input_size_）可走直接 DMA-FD 路径
   * - BGR888/NV12/NV21/RGBA/BGRA 会回退到拷贝+颜色转换路径
   *
   * @param input 包含预处理图像的 DMA-BUF
   * @param original_size 原图尺寸
   * @param letterbox_info letterbox 映射参数
   * @return 检测结果（原图坐标系）
   */
  std::vector<Detection> inferDmaBuf(
      rkapp::common::DmaBuf& input,
      const cv::Size& original_size,
      const struct rkapp::preprocess::LetterboxInfo& letterbox_info);

  // 设置 NPU 核心掩码（例如 0x1/0x2/0x4/0x3/0x7）。
  void setCoreMask(uint32_t core_mask);
  void setDecodeParams(const DecodeParams& params) override;

  // 线程安全地读取推理时推断出的解码元信息。
  int num_classes() const;
  bool has_objness() const;
  std::vector<std::string> class_names() const;

  int getInputWidth() const override;
  int getInputHeight() const override;

private:
  struct Impl;
  // 保护下列共享状态（init/infer/release 可能来自不同线程）。
  mutable std::mutex state_mutex_;
  std::shared_ptr<Impl> impl_;
  std::string model_path_;
  int input_size_ = 640;     // 推理输入边长（方形）
  bool is_initialized_ = false;
  uint32_t core_mask_ = 0x7; // 默认三核并行；可被 setCoreMask 覆盖
  
  // 自动推断/缓存的解码参数（初始化成功后由 metadata 和输出张量共同确定）
  int num_classes_ = -1;
  bool has_objness_ = true;  // Most YOLO exports include objectness score
  std::vector<std::string> class_names_;
  DecodeParams decode_params_;
  ModelMeta model_meta_;
};

} // namespace rkapp::infer

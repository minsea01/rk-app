#include "rkapp/capture/GigeSource.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#ifdef RKAPP_WITH_GIGE
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
// file-scope holders for a simple single-stream implementation
static GstElement* g_gige_pipeline = nullptr;
static GstElement* g_gige_sink = nullptr;
#endif

namespace rkapp::capture {

bool GigeSource::open(const std::string& uri) {
#ifdef RKAPP_WITH_GIGE
  // 使用GStreamer aravissrc + appsink 作为GigE采集实现
  // 支持的uri形式：camera-name=<id_or_first>[,width=W,height=H,framerate=F]
  // 例如: "camera-name=Aravis-Fake-GV01,width=512,height=512,framerate=25/1"
  if (!gst_is_initialized()) {
    int argc = 0; char** argv = nullptr; gst_init(&argc, &argv);
  }
  
  // 解析URI参数
  std::string camera_name, caps_filter;
  size_t pos = uri.find("camera-name=");
  if (pos != std::string::npos) {
    size_t start = pos + 12; // "camera-name="的长度
    size_t comma = uri.find(",", start);
    if (comma != std::string::npos) {
      camera_name = uri.substr(start, comma - start);
      // 解析其他参数并构建caps
      std::string params = uri.substr(comma + 1);
      // 若未显式指定format，默认灰度以降低带宽与CPU负载
      if (params.find("format=") == std::string::npos) {
        params = std::string("format=GRAY8,") + params;
      }
      caps_filter = "video/x-raw," + params;
    } else {
      camera_name = uri.substr(start);
      // 默认灰度
      caps_filter = "video/x-raw,format=GRAY8";
    }
  } else {
    camera_name = uri;
    caps_filter = "video/x-raw,format=GRAY8"; // 默认灰度
  }

  // 如果用户已指定 format=GRAY8，则避免多余的 videoconvert->BGR，直接推给 appsink
  bool want_gray8 = (caps_filter.find("format=GRAY8") != std::string::npos);
  std::string pipeline_desc;
  if (want_gray8) {
    pipeline_desc = "aravissrc camera-name=\"" + camera_name + "\" ! " + caps_filter +
                    " ! appsink name=sink caps=video/x-raw,format=GRAY8 sync=false max-buffers=2 drop=true";
  } else {
    pipeline_desc = "aravissrc camera-name=\"" + camera_name + "\" ! " + caps_filter +
                    " ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink sync=false max-buffers=2 drop=true";
  }
  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &err);
  if (!pipeline) {
    std::cerr << "GigeSource: 创建GStreamer管道失败: " << (err?err->message:"unknown") << std::endl;
    if (err) g_error_free(err);
    opened_ = false; return false;
  }
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
  if (!sink) { std::cerr << "GigeSource: 未找到appsink" << std::endl; gst_object_unref(pipeline); opened_ = false; return false; }
  gst_app_sink_set_emit_signals((GstAppSink*)sink, false);
  gst_app_sink_set_drop((GstAppSink*)sink, true);
  gst_app_sink_set_max_buffers((GstAppSink*)sink, 2);
  std::cout << "GigeSource: 尝试启动管道: " << pipeline_desc << std::endl;
  GstStateChangeReturn sret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (sret == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "GigeSource: 管道无法进入PLAYING" << std::endl;
    // 获取更详细的错误信息
    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
    if (msg) {
      GError* err; gchar* debug_info;
      gst_message_parse_error(msg, &err, &debug_info);
      std::cerr << "GStreamer错误: " << err->message << std::endl;
      if (debug_info) std::cerr << "调试信息: " << debug_info << std::endl;
      g_error_free(err); g_free(debug_info); gst_message_unref(msg);
    }
    gst_object_unref(bus);
    gst_object_unref(sink); gst_object_unref(pipeline); opened_ = false; return false;
  }
  // 暂存到全局静态（简单实现）：
  g_gige_pipeline = pipeline; g_gige_sink = sink;
  opened_ = true; size_ = {0,0}; fps_ = 30.0; count_ = 0;
  return true;
#else
  std::cerr << "GigeSource: 未启用 GIGE (Aravis)。请以 -DENABLE_GIGE=ON 并安装 aravis-0.8 后重编译。" << std::endl;
  opened_ = false;
  return false;
#endif
}

bool GigeSource::read(cv::Mat& frame) {
  if (!opened_) return false;
#ifdef RKAPP_WITH_GIGE
  if (!g_gige_sink) return false;
  GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(g_gige_sink));
  if (!sample) return false;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  GstStructure const* s = gst_caps_get_structure(caps, 0);
  int width=0,height=0; gst_structure_get_int(s, "width", &width); gst_structure_get_int(s, "height", &height);
  const gchar* fmt = gst_structure_get_string(s, "format");
  GstMapInfo map; gst_buffer_map(buffer, &map, GST_MAP_READ);
  if (fmt && g_strcmp0(fmt, "GRAY8") == 0) {
    cv::Mat img(height, width, CV_8UC1, (void*)map.data);
    img.copyTo(frame);
  } else {
    cv::Mat img(height, width, CV_8UC3, (void*)map.data);
    img.copyTo(frame);
  }
  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
  if (size_.width==0) size_ = {width,height};
  ++count_;
  return true;
#else
  return false;
#endif
}

void GigeSource::release() { opened_ = false; count_ = 0; }
bool GigeSource::isOpened() const { return opened_; }
double GigeSource::getFPS() const { return fps_; }
cv::Size GigeSource::getSize() const { return size_; }
int GigeSource::getTotalFrames() const { return 0; }
int GigeSource::getCurrentFrame() const { return count_; }

} // namespace rkapp::capture


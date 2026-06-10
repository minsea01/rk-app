#pragma once

#include "rkapp/output/IOutput.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <netinet/in.h>
#include <mutex>
#include <thread>

namespace rkapp::output {

// TCP/文件结果输出。
//
// 线程模型：send() 仅把结果放入有界队列（满则丢最旧）并立即返回；
// 序列化(JSON+base64)、文件写入、socket 发送与重连全部在内部发送线程执行，
// 不占用调用方（通常是推理管线）线程的时间。
class TcpOutput : public IOutput {
public:
  TcpOutput();
  ~TcpOutput() override;

  bool open(const std::string& config = "") override;
  // 异步入队；返回 true 表示已接受待发（不代表已送达）。
  // 队列满时丢弃最旧条目并计入 droppedFrames()。
  bool send(const FrameResult& result) override;
  void close() override;
  bool isOpened() const override;

  OutputType getType() const override;
  bool isConnected() const;
  size_t backlogDepth() const;
  std::chrono::milliseconds reconnectBackoff() const;

private:
  struct QueuedPayload {
    std::string data;
    size_t offset = 0;
  };

  void senderLoop();
  // 处理单个结果：序列化 -> 文件写入 -> TCP 发送（已连接时）。
  // allow_tcp_retry=false 用于关闭阶段的有界收尾。
  void processItem(const FrameResult& result, bool allow_tcp_retry);
  bool setup_socket();
  bool setup_socket_locked();
  bool attemptReconnect();
  bool sendBuffer(QueuedPayload& payload);
  void closeSocket();
  void closeSocketLocked();

  std::string server_ip_ = "127.0.0.1";
  int server_port_ = 9000;
  int socket_fd_ = -1;
  struct sockaddr_in server_addr_{};
  std::atomic<bool> tcp_connected_{false};
  std::atomic<bool> is_opened_{false};
  bool endpoint_configured_ = false;

  // Optional NIC/source binding
  // - bind_interface_: try SO_BINDTODEVICE (requires CAP_NET_RAW/root). Example: "eth1"
  // - bind_ip_: bind local source address before connect(). Example: "10.0.0.100"
  std::string bind_interface_;
  std::string bind_ip_;

  // File output（仅发送线程访问；open/close 在线程未运行时访问）
  bool enable_file_output_ = false;
  std::string file_path_;
  std::ofstream file_output_;

  std::chrono::steady_clock::time_point last_reconnect_attempt_{};
  std::atomic<bool> has_reconnect_attempt_{false};
  std::chrono::milliseconds reconnect_backoff_initial_{500};
  std::chrono::milliseconds reconnect_backoff_{500};
  std::chrono::milliseconds reconnect_backoff_max_{5000};

  // 待发队列与发送线程
  std::deque<FrameResult> pending_;
  size_t max_backlog_ = 64;
  mutable std::mutex queue_mtx_;
  std::condition_variable queue_cv_;
  bool stop_requested_ = false;          // guarded by queue_mtx_
  std::atomic<bool> stop_flag_{false};   // 供发送循环内的快速检查
  std::thread sender_thread_;
  mutable std::mutex socket_mtx_;

  // Statistics for monitoring
  std::atomic<uint64_t> dropped_frames_{0};
  std::atomic<uint64_t> total_sent_{0};

public:
  /**
   * @brief Get number of frames dropped (queue overflow or connection lost mid-send)
   */
  uint64_t droppedFrames() const { return dropped_frames_.load(std::memory_order_relaxed); }

  /**
   * @brief Get total number of frames successfully sent
   */
  uint64_t totalSent() const { return total_sent_.load(std::memory_order_relaxed); }

  /**
   * @brief Reset statistics counters
   */
  void resetStats() {
    dropped_frames_.store(0, std::memory_order_relaxed);
    total_sent_.store(0, std::memory_order_relaxed);
  }
};

} // namespace rkapp::output

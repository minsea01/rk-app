#pragma once

#include "rkapp/output/IOutput.hpp"
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <netinet/in.h>
#include <mutex>

namespace rkapp::output {

class TcpOutput : public IOutput {
public:
  TcpOutput();
  ~TcpOutput() override;

  bool open(const std::string& config = "") override;
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

  bool setup_socket();
  bool attemptReconnect();
  bool flushBacklog();
  bool sendBuffer(QueuedPayload& payload);
  void closeSocket();

  std::string server_ip_ = "127.0.0.1";
  int server_port_ = 9000;
  int socket_fd_ = -1;
  struct sockaddr_in server_addr_{};
  std::atomic<bool> tcp_connected_{false};
  bool is_opened_ = false;
  bool endpoint_configured_ = false;

  // Optional NIC/source binding
  // - bind_interface_: try SO_BINDTODEVICE (requires CAP_NET_RAW/root). Example: "eth1"
  // - bind_ip_: bind local source address before connect(). Example: "10.0.0.100"
  std::string bind_interface_;
  std::string bind_ip_;

  // File output
  bool enable_file_output_ = false;
  std::string file_path_;
  std::ofstream file_output_;

  std::chrono::steady_clock::time_point last_reconnect_attempt_{};
  bool has_reconnect_attempt_ = false;
  std::chrono::milliseconds reconnect_backoff_initial_{500};
  std::chrono::milliseconds reconnect_backoff_{500};
  std::chrono::milliseconds reconnect_backoff_max_{5000};

  std::deque<QueuedPayload> backlog_;
  size_t max_backlog_ = 64;
  mutable std::mutex backlog_mtx_;

  // Statistics for monitoring
  std::atomic<uint64_t> dropped_frames_{0};
  std::atomic<uint64_t> total_sent_{0};

public:
  /**
   * @brief Get number of frames dropped due to backlog overflow
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

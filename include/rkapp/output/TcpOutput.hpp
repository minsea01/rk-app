#pragma once

#include "rkapp/output/IOutput.hpp"
#include <chrono>
#include <fstream>
#include <netinet/in.h>

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

private:
  bool setup_socket();
  bool attemptReconnect();
  void closeSocket();

  std::string server_ip_ = "127.0.0.1";
  int server_port_ = 9000;
  int socket_fd_ = -1;
  struct sockaddr_in server_addr_{};
  bool tcp_connected_ = false;
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
};

} // namespace rkapp::output

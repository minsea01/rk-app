#include "rkapp/output/TcpOutput.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <fcntl.h>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <net/if.h>
#include <netinet/tcp.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <system_error>

#include "log.hpp"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace rkapp::output {

TcpOutput::TcpOutput() = default;
TcpOutput::~TcpOutput() { close(); }

namespace {

std::string escape_json(const std::string& value) {
        std::string escaped;
        escaped.reserve(value.size() + 8);
        for (char ch : value) {
            switch (ch) {
                case '\\': escaped += "\\\\"; break;
                case '"': escaped += "\\\""; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                default: {
                    const unsigned char c = static_cast<unsigned char>(ch);
                    if (c < 0x20) {
                        constexpr char hex[] = "0123456789ABCDEF";
                        escaped += "\\u00";
                        escaped += hex[(c >> 4) & 0x0F];
                        escaped += hex[c & 0x0F];
                    } else {
                        escaped += ch;
                    }
                } break;
            }
        }
        return escaped;
}

}  // namespace

bool TcpOutput::open(const std::string& config) {
        close();

        enable_file_output_ = false;
        {
            std::lock_guard<std::mutex> lock(backlog_mtx_);
            backlog_.clear();
        }
        file_path_.clear();
        bind_interface_.clear();
        bind_ip_.clear();
        has_reconnect_attempt_ = false;
        endpoint_configured_ = true;
        reconnect_backoff_initial_ = std::chrono::milliseconds(500);
        reconnect_backoff_max_ = std::chrono::milliseconds(5000);
        reconnect_backoff_ = reconnect_backoff_initial_;
        max_backlog_ = 64;
        last_reconnect_attempt_ = {};

        server_ip_ = "127.0.0.1";
        server_port_ = 9000;

        std::istringstream ss(config);
        std::string part;

        if (!config.empty()) {
            if (!std::getline(ss, part, ',')) {
                LOGE("TcpOutput: empty config string");
                return false;
            }

            const auto colon_pos = part.find(':');
            if (colon_pos == std::string::npos) {
                LOGE("TcpOutput: invalid endpoint format: ", part);
                return false;
            }

            server_ip_ = part.substr(0, colon_pos);
            const std::string port_str = part.substr(colon_pos + 1);
            int port = 0;
            const auto res = std::from_chars(port_str.data(), port_str.data() + port_str.size(), port);
            if (res.ec != std::errc{} || port <= 0 || port > 65535) {
                LOGE("TcpOutput: invalid port in config: ", port_str);
                return false;
            }
            server_port_ = port;
        }

        while (std::getline(ss, part, ',')) {
            if (part.empty()) continue;
            if (part.rfind("file:", 0) == 0) {
                file_path_ = part.substr(5);
                enable_file_output_ = !file_path_.empty();
            } else if (part.rfind("iface:", 0) == 0) {
                bind_interface_ = part.substr(6);
            } else if (part.rfind("bind_ip:", 0) == 0) {
                bind_ip_ = part.substr(8);
            } else if (part.rfind("queue:", 0) == 0) {
                const std::string depth_str = part.substr(6);
                int depth = 0;
                const auto res = std::from_chars(depth_str.data(), depth_str.data() + depth_str.size(), depth);
                if (res.ec == std::errc{} && depth > 0) {
                    max_backlog_ = static_cast<size_t>(depth);
                } else {
                    LOGW("TcpOutput: invalid queue depth '", depth_str, "', keeping default");
                }
            } else if (part.rfind("backoff:", 0) == 0) {
                const std::string backoff_str = part.substr(8);
                int backoff = 0;
                const auto res = std::from_chars(backoff_str.data(),
                                                 backoff_str.data() + backoff_str.size(), backoff);
                if (res.ec == std::errc{} && backoff > 0) {
                    backoff = std::clamp(backoff, 50, 10000);
                    reconnect_backoff_initial_ = std::chrono::milliseconds(backoff);
                    reconnect_backoff_ = reconnect_backoff_initial_;
                } else {
                    LOGW("TcpOutput: invalid backoff '", backoff_str, "', keeping default");
                }
            } else if (part.rfind("backoff_max:", 0) == 0) {
                const std::string backoff_str = part.substr(12);
                int backoff = 0;
                const auto res = std::from_chars(backoff_str.data(),
                                                 backoff_str.data() + backoff_str.size(), backoff);
                if (res.ec == std::errc{} && backoff > 0) {
                    backoff = std::clamp(backoff, 100, 60000);
                    reconnect_backoff_max_ = std::chrono::milliseconds(backoff);
                } else {
                    LOGW("TcpOutput: invalid backoff_max '", backoff_str, "', keeping default");
                }
            } else {
                LOGW("TcpOutput: unknown option '", part, "' (ignored)");
            }
        }

        if (reconnect_backoff_max_ < reconnect_backoff_initial_) {
            reconnect_backoff_max_ = reconnect_backoff_initial_;
        }

        if (enable_file_output_) {
            file_output_.open(file_path_, std::ios::app);
            if (!file_output_.is_open()) {
                LOGE("TcpOutput: failed to open output file: ", file_path_);
                enable_file_output_ = false;
            } else {
                LOGI("TcpOutput: logging results to ", file_path_);
            }
        }

        if (endpoint_configured_) {
            if (!setup_socket()) {
                LOGW("TcpOutput: initial connect failed, will retry automatically");
            }
        }

        is_opened_ = endpoint_configured_ || enable_file_output_;
        if (is_opened_) {
            last_reconnect_attempt_ = std::chrono::steady_clock::now() - reconnect_backoff_;
        }
        return is_opened_;
}

bool TcpOutput::send(const FrameResult& result) {
        if (!is_opened_) {
            return false;
        }

        if (endpoint_configured_ && !tcp_connected_.load()) {
            attemptReconnect();
        }

        std::ostringstream json;
        json << '{'
             << "\"frame_id\":" << result.frame_id << ','
             << "\"timestamp\":" << result.timestamp << ','
             << "\"width\":" << result.width << ','
             << "\"height\":" << result.height << ','
             << "\"source_uri\":\"" << escape_json(result.source_uri) << "\",";

        json << "\"detections\":[";
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& det = result.detections[i];
            if (i > 0) json << ',';
            json << '{'
                 << "\"x\":" << det.x << ','
                 << "\"y\":" << det.y << ','
                 << "\"w\":" << det.w << ','
                 << "\"h\":" << det.h << ','
                 << "\"confidence\":" << det.confidence << ','
                 << "\"class_id\":" << det.class_id << ','
                 << "\"class_name\":\"" << escape_json(det.class_name) << "\"";
            json << '}';
        }
        json << "]}\n";

        const std::string payload = json.str();

        if (enable_file_output_ && file_output_.is_open()) {
            file_output_ << payload;
            file_output_.flush();
        }

        {
            std::lock_guard<std::mutex> lock(backlog_mtx_);
            backlog_.push_back(QueuedPayload{payload, 0});
            if (backlog_.size() > max_backlog_) {
                LOGW("TcpOutput: backlog full (max=", max_backlog_, "), dropping oldest frame");
                backlog_.pop_front();
            }
        }

        if (!tcp_connected_.load()) {
            attemptReconnect();
        }

        bool delivered = tcp_connected_.load() && flushBacklog();

        return delivered || (enable_file_output_ && file_output_.is_open());
}

void TcpOutput::close() {
        closeSocket();

        if (file_output_.is_open()) {
            file_output_.close();
        }

        {
            std::lock_guard<std::mutex> lock(backlog_mtx_);
            backlog_.clear();
        }
        has_reconnect_attempt_ = false;
        last_reconnect_attempt_ = {};
        reconnect_backoff_ = reconnect_backoff_initial_;
        is_opened_ = false;
}

bool TcpOutput::isOpened() const { return is_opened_; }

OutputType TcpOutput::getType() const { return OutputType::TCP; }

bool TcpOutput::isConnected() const { return tcp_connected_.load(); }

size_t TcpOutput::backlogDepth() const {
        std::lock_guard<std::mutex> lock(backlog_mtx_);
        return backlog_.size();
}

std::chrono::milliseconds TcpOutput::reconnectBackoff() const {
        return reconnect_backoff_;
}

void TcpOutput::closeSocket() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        tcp_connected_.store(false);
}

bool TcpOutput::setup_socket() {
        if (!endpoint_configured_) {
            return false;
        }

        closeSocket();

        socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            LOGE("TcpOutput: failed to create socket");
            return false;
        }

        int flag = 1;
        if (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) != 0) {
            LOGW("TcpOutput: TCP_NODELAY failed (errno=", errno, ")");
        }

        if (!bind_ip_.empty()) {
            sockaddr_in local{};
            local.sin_family = AF_INET;
            local.sin_port = 0;
            if (inet_pton(AF_INET, bind_ip_.c_str(), &local.sin_addr) <= 0) {
                LOGW("TcpOutput: invalid bind_ip=", bind_ip_);
            } else if (::bind(socket_fd_, reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0) {
                LOGW("TcpOutput: bind(bind_ip) failed (errno=", errno, ")");
            } else {
                LOGI("TcpOutput: bound local ip ", bind_ip_);
            }
        }

        if (!bind_interface_.empty()) {
            if (bind_interface_.size() >= IFNAMSIZ) {
                LOGW("TcpOutput: iface name too long: ", bind_interface_);
            } else {
                struct ifreq ifr;
                std::memset(&ifr, 0, sizeof(ifr));
                std::snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", bind_interface_.c_str());
                if (setsockopt(socket_fd_, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr)) != 0) {
                    LOGW("TcpOutput: SO_BINDTODEVICE failed (errno=", errno, ")");
                } else {
                    LOGI("TcpOutput: bound to iface ", bind_interface_);
                }
            }
        }

        server_addr_ = {};
        server_addr_.sin_family = AF_INET;
        server_addr_.sin_port = htons(static_cast<uint16_t>(server_port_));
        if (inet_pton(AF_INET, server_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
            LOGE("TcpOutput: invalid server ip ", server_ip_);
            closeSocket();
            return false;
        }

        if (::connect(socket_fd_, reinterpret_cast<sockaddr*>(&server_addr_), sizeof(server_addr_)) == 0) {
            tcp_connected_.store(true);
            has_reconnect_attempt_ = false;
            reconnect_backoff_ = reconnect_backoff_initial_;
            LOGI("TcpOutput: connected to ", server_ip_, ":", server_port_);

            if (const char* env_snd = std::getenv("RKAPP_TCP_SNDBUF")) {
                const int sz = std::atoi(env_snd);
                if (sz > 0 && setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz)) == 0) {
                    LOGI("TcpOutput: SO_SNDBUF set to ", sz);
                }
            }

            int flags = fcntl(socket_fd_, F_GETFL, 0);
            if (flags >= 0) {
                fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
            }
        } else {
            LOGW("TcpOutput: connect to ", server_ip_, ":", server_port_,
                 " failed (errno=", errno, ")");
            tcp_connected_.store(false);
            closeSocket();
            return false;
        }

        return true;
}

bool TcpOutput::attemptReconnect() {
        const auto now = std::chrono::steady_clock::now();
        if (has_reconnect_attempt_ &&
            now - last_reconnect_attempt_ < reconnect_backoff_) {
            return tcp_connected_.load();
        }

        last_reconnect_attempt_ = now;
        has_reconnect_attempt_ = true;
        if (setup_socket()) {
            flushBacklog();
            return true;
        }
        reconnect_backoff_ = std::min(reconnect_backoff_ * 2, reconnect_backoff_max_);
        return false;
}

bool TcpOutput::flushBacklog() {
        bool delivered_any = false;
        while (tcp_connected_.load()) {
            QueuedPayload current;
            {
                std::lock_guard<std::mutex> lock(backlog_mtx_);
                if (backlog_.empty()) {
                    break;
                }
                current = backlog_.front();
            }

            bool sent = sendBuffer(current);

            {
                std::lock_guard<std::mutex> lock(backlog_mtx_);
                if (!backlog_.empty()) {
                    if (sent) {
                        backlog_.pop_front();
                        delivered_any = true;
                    } else {
                        backlog_.front().offset = current.offset;
                    }
                }
            }

            if (!sent || !tcp_connected_.load()) {
                break;
            }
        }
        return delivered_any;
}

bool TcpOutput::sendBuffer(QueuedPayload& payload) {
        if (!tcp_connected_.load()) return false;

        while (payload.offset < payload.data.size()) {
            const ssize_t sent = ::send(socket_fd_,
                                        payload.data.data() + payload.offset,
                                        payload.data.size() - payload.offset,
                                        MSG_NOSIGNAL);
            if (sent > 0) {
                payload.offset += static_cast<size_t>(sent);
                continue;
            }

            if (sent < 0 && errno == EINTR) {
                continue;
            }

            if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                return false;
            }

            LOGW("TcpOutput: send failed (errno=", errno, ")");
            closeSocket();
            return false;
        }

        return true;
}

} // namespace rkapp::output

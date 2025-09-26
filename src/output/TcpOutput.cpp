#include "rkapp/output/TcpOutput.hpp"

#include <arpa/inet.h>
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

        // Reset optional fields
        enable_file_output_ = false;
        file_path_.clear();
        bind_interface_.clear();
        bind_ip_.clear();
        endpoint_configured_ = false;
        has_reconnect_attempt_ = false;

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

        endpoint_configured_ = true;

        while (std::getline(ss, part, ',')) {
            if (part.rfind("file:", 0) == 0) {
                file_path_ = part.substr(5);
                enable_file_output_ = !file_path_.empty();
            } else if (part.rfind("iface:", 0) == 0) {
                bind_interface_ = part.substr(6);
            } else if (part.rfind("bind_ip:", 0) == 0) {
                bind_ip_ = part.substr(8);
            } else if (!part.empty()) {
                LOGW("TcpOutput: unknown option '", part, "' (ignored)");
            }
        }

        setup_socket();

        if (enable_file_output_) {
            file_output_.open(file_path_, std::ios::app);
            if (!file_output_.is_open()) {
                LOGE("TcpOutput: failed to open output file: ", file_path_);
                enable_file_output_ = false;
            } else {
                LOGI("TcpOutput: logging results to ", file_path_);
            }
        }

        is_opened_ = tcp_connected_ || enable_file_output_;
        return is_opened_;
}

bool TcpOutput::send(const FrameResult& result) {
        if (!is_opened_) {
            return false;
        }

        if (endpoint_configured_ && !tcp_connected_) {
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
        bool delivered = false;

        if (tcp_connected_) {
            const char* data = payload.data();
            size_t remaining = payload.size();
            while (remaining > 0) {
                const ssize_t sent = ::send(socket_fd_, data, remaining, 0);
                if (sent > 0) {
                    data += sent;
                    remaining -= static_cast<size_t>(sent);
                } else if (sent < 0 && errno == EINTR) {
                    continue;
                } else {
                    LOGW("TcpOutput: send failed (errno=", errno, ")");
                    tcp_connected_ = false;
                    attemptReconnect();
                    break;
                }
            }
            delivered = (remaining == 0);
        }

        if (enable_file_output_ && file_output_.is_open()) {
            file_output_ << payload;
            file_output_.flush();
            delivered = true;
        }

        return delivered;
}

void TcpOutput::close() {
        closeSocket();

        if (file_output_.is_open()) {
            file_output_.close();
        }

        is_opened_ = false;
}

bool TcpOutput::isOpened() const { return is_opened_; }

OutputType TcpOutput::getType() const { return OutputType::TCP; }

void TcpOutput::closeSocket() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        tcp_connected_ = false;
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
            tcp_connected_ = true;
            has_reconnect_attempt_ = false;
            LOGI("TcpOutput: connected to ", server_ip_, ":", server_port_);

            if (const char* env_snd = std::getenv("RKAPP_TCP_SNDBUF")) {
                const int sz = std::atoi(env_snd);
                if (sz > 0 && setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz)) == 0) {
                    LOGI("TcpOutput: SO_SNDBUF set to ", sz);
                }
            }
        } else {
            LOGW("TcpOutput: connect to ", server_ip_, ":", server_port_,
                 " failed (errno=", errno, ")");
            tcp_connected_ = false;
        }

        return true;
}

bool TcpOutput::attemptReconnect() {
        const auto now = std::chrono::steady_clock::now();
        if (has_reconnect_attempt_ &&
            now - last_reconnect_attempt_ < std::chrono::seconds(1)) {
            return tcp_connected_;
        }

        last_reconnect_attempt_ = now;
        has_reconnect_attempt_ = true;
        return setup_socket();
}

} // namespace rkapp::output

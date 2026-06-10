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
#include <net/if.h>
#include <netinet/tcp.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <system_error>
#include <poll.h>

#include "rkapp/common/log.hpp"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace rkapp::output {

TcpOutput::TcpOutput() = default;
TcpOutput::~TcpOutput() { close(); }

namespace {

// NDJSON 协议版本：接收端可据此做兼容处理；字段新增时保持向后兼容，
// 字段语义变化时递增版本号。
constexpr int kProtocolVersion = 1;

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
                    if (c < 0x20 || c > 0x7E) {
                        // Escape control characters and non-ASCII bytes as \u00XX.
                        // Avoids embedding raw non-UTF-8 bytes in the JSON stream.
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

std::string base64_encode(const std::vector<uint8_t>& bytes) {
        static constexpr char kAlphabet[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (bytes.empty()) {
            return {};
        }

        std::string out;
        out.reserve(((bytes.size() + 2) / 3) * 4);
        size_t i = 0;
        while (i + 2 < bytes.size()) {
            const uint32_t block = (static_cast<uint32_t>(bytes[i]) << 16) |
                                   (static_cast<uint32_t>(bytes[i + 1]) << 8) |
                                   static_cast<uint32_t>(bytes[i + 2]);
            out.push_back(kAlphabet[(block >> 18) & 0x3F]);
            out.push_back(kAlphabet[(block >> 12) & 0x3F]);
            out.push_back(kAlphabet[(block >> 6) & 0x3F]);
            out.push_back(kAlphabet[block & 0x3F]);
            i += 3;
        }

        if (i < bytes.size()) {
            uint32_t block = static_cast<uint32_t>(bytes[i]) << 16;
            out.push_back(kAlphabet[(block >> 18) & 0x3F]);
            if (i + 1 < bytes.size()) {
                block |= static_cast<uint32_t>(bytes[i + 1]) << 8;
                out.push_back(kAlphabet[(block >> 12) & 0x3F]);
                out.push_back(kAlphabet[(block >> 6) & 0x3F]);
                out.push_back('=');
            } else {
                out.push_back(kAlphabet[(block >> 12) & 0x3F]);
                out.push_back('=');
                out.push_back('=');
            }
        }

        return out;
}

// 把单帧结果序列化为一行 NDJSON。仅在发送线程调用。
std::string serializePayload(const FrameResult& result) {
        std::ostringstream json;
        const auto us_to_ms = [](int64_t us) -> double {
            return static_cast<double>(us) / 1000.0;
        };

        const int64_t process_us = result.timing.processUs();
        json << '{'
             << "\"v\":" << kProtocolVersion << ','
             << "\"frame_id\":" << result.frame_id << ','
             << "\"timestamp\":" << result.timestamp << ','
             << "\"width\":" << result.width << ','
             << "\"height\":" << result.height << ','
             << "\"source_uri\":\"" << escape_json(result.source_uri) << "\","
             << "\"latency_ms\":" << us_to_ms(process_us) << ',';

        json << "\"timing\":{"
             << "\"capture_wait_ms\":" << us_to_ms(result.timing.capture_us) << ','
             << "\"preprocess_ms\":" << us_to_ms(result.timing.preprocess_us) << ','
             << "\"inference_ms\":" << us_to_ms(result.timing.inference_us) << ','
             << "\"postprocess_ms\":" << us_to_ms(result.timing.postprocess_us) << ','
             << "\"process_ms\":" << us_to_ms(process_us) << ','
             << "\"total_with_capture_wait_ms\":" << us_to_ms(result.timing.total_us)
             << "},";

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
        json << ']';

        if (!result.image_bytes.empty()) {
            json << ",\"image\":{"
                 << "\"encoding\":\"" << escape_json(result.image_encoding.empty()
                                                        ? std::string("jpeg")
                                                        : result.image_encoding)
                 << "\","
                 << "\"contains_overlays\":"
                 << (result.image_contains_overlays ? "true" : "false") << ','
                 << "\"width\":" << result.image_width << ','
                 << "\"height\":" << result.image_height << ','
                 << "\"roi_applied\":"
                 << (result.image_roi_applied ? "true" : "false") << ','
                 << "\"roi\":{"
                 << "\"x\":" << result.image_roi.x << ','
                 << "\"y\":" << result.image_roi.y << ','
                 << "\"w\":" << result.image_roi.width << ','
                 << "\"h\":" << result.image_roi.height
                 << "},"
                 << "\"data_base64\":\"" << base64_encode(result.image_bytes) << "\""
                 << '}';
        }

        json << "}\n";
        return json.str();
}

}  // namespace

bool TcpOutput::open(const std::string& config) {
        close();

        enable_file_output_ = false;
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            pending_.clear();
        }
        dropped_frames_.store(0, std::memory_order_relaxed);
        total_sent_.store(0, std::memory_order_relaxed);
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
            file_output_.open(file_path_, std::ios::out | std::ios::trunc);
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

        is_opened_.store(endpoint_configured_ || enable_file_output_);
        if (is_opened_.load()) {
            {
                std::lock_guard<std::mutex> socket_lock(socket_mtx_);
                last_reconnect_attempt_ = std::chrono::steady_clock::now() - reconnect_backoff_;
            }
            {
                std::lock_guard<std::mutex> lock(queue_mtx_);
                stop_requested_ = false;
            }
            stop_flag_.store(false);
            sender_thread_ = std::thread(&TcpOutput::senderLoop, this);
        }
        return is_opened_.load();
}

bool TcpOutput::send(const FrameResult& result) {
        if (!is_opened_.load()) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            if (stop_requested_) {
                return false;
            }
            pending_.push_back(result);
            if (pending_.size() > max_backlog_) {
                pending_.pop_front();
                const uint64_t dropped =
                    dropped_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
                LOGW("TcpOutput: backlog full (max=", max_backlog_,
                     "), dropping oldest frame (total dropped: ", dropped, ")");
            }
        }
        queue_cv_.notify_one();
        return true;
}

void TcpOutput::senderLoop() {
        for (;;) {
            FrameResult item;
            {
                std::unique_lock<std::mutex> lock(queue_mtx_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(250), [&] {
                    return stop_requested_ || !pending_.empty();
                });
                if (stop_requested_) {
                    break;  // 剩余条目在循环外做有界收尾
                }
                if (pending_.empty()) {
                    continue;
                }

                // 未连接且无文件输出时不出队：条目保留在有界窗口内等待重连成功。
                if (!tcp_connected_.load() && !enable_file_output_) {
                    lock.unlock();
                    if (endpoint_configured_) {
                        attemptReconnect();
                    }
                    if (!tcp_connected_.load()) {
                        std::unique_lock<std::mutex> idle_lock(queue_mtx_);
                        queue_cv_.wait_for(idle_lock, std::chrono::milliseconds(100),
                                           [&] { return stop_requested_; });
                    }
                    continue;
                }

                item = std::move(pending_.front());
                pending_.pop_front();
            }

            processItem(item, /*allow_tcp_retry=*/true);

            if (endpoint_configured_ && !tcp_connected_.load() && !stop_flag_.load()) {
                attemptReconnect();
            }
        }

        // 关闭收尾：剩余条目仍写入文件（若启用）；TCP 仅在已连接时做单次有界尝试。
        std::deque<FrameResult> rest;
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            rest.swap(pending_);
        }
        for (const auto& result : rest) {
            processItem(result, /*allow_tcp_retry=*/false);
        }
}

void TcpOutput::processItem(const FrameResult& result, bool allow_tcp_retry) {
        std::string payload = serializePayload(result);

        if (enable_file_output_ && file_output_.is_open()) {
            file_output_ << payload;
            file_output_.flush();
        }

        if (!endpoint_configured_ || !tcp_connected_.load()) {
            return;
        }

        QueuedPayload inflight{std::move(payload), 0};
        bool delivered = sendBuffer(inflight);
        // sendBuffer 在内核缓冲满(EAGAIN)时短暂 poll 后返回 false 且保留 offset，
        // 连接仍在则继续推进同一载荷；硬错误时它会断开连接使循环退出。
        while (!delivered && allow_tcp_retry && tcp_connected_.load() && !stop_flag_.load()) {
            delivered = sendBuffer(inflight);
        }

        if (delivered) {
            total_sent_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // 连接中断或关闭收尾未送完：放弃该帧，保证每个连接内的 NDJSON 行边界完整
            // （不跨连接续传半行，避免接收端解析到脏数据）。
            const uint64_t dropped = dropped_frames_.fetch_add(1, std::memory_order_relaxed) + 1;
            LOGW("TcpOutput: frame not delivered (connection lost mid-send), dropped total: ",
                 dropped);
        }
}

void TcpOutput::close() {
        is_opened_.store(false);
        stop_flag_.store(true);
        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            stop_requested_ = true;
        }
        queue_cv_.notify_all();
        if (sender_thread_.joinable()) {
            sender_thread_.join();
        }

        {
            std::lock_guard<std::mutex> socket_lock(socket_mtx_);
            closeSocketLocked();
            has_reconnect_attempt_ = false;
            last_reconnect_attempt_ = {};
            reconnect_backoff_ = reconnect_backoff_initial_;
        }

        if (file_output_.is_open()) {
            file_output_.close();
        }

        {
            std::lock_guard<std::mutex> lock(queue_mtx_);
            pending_.clear();
            stop_requested_ = false;  // 允许再次 open()
        }
        stop_flag_.store(false);
}

bool TcpOutput::isOpened() const { return is_opened_.load(); }

OutputType TcpOutput::getType() const { return OutputType::TCP; }

bool TcpOutput::isConnected() const { return tcp_connected_.load(); }

size_t TcpOutput::backlogDepth() const {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        return pending_.size();
}

std::chrono::milliseconds TcpOutput::reconnectBackoff() const {
        std::lock_guard<std::mutex> lock(socket_mtx_);
        return reconnect_backoff_;
}

void TcpOutput::closeSocket() {
        std::lock_guard<std::mutex> lock(socket_mtx_);
        closeSocketLocked();
}

void TcpOutput::closeSocketLocked() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        tcp_connected_.store(false);
}

bool TcpOutput::setup_socket() {
        std::lock_guard<std::mutex> lock(socket_mtx_);
        return setup_socket_locked();
}

bool TcpOutput::setup_socket_locked() {
        if (!endpoint_configured_) {
            return false;
        }

        closeSocketLocked();

        socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            LOGE("TcpOutput: failed to create socket");
            return false;
        }

        int flags = fcntl(socket_fd_, F_GETFL, 0);
        if (flags >= 0) {
            fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
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
            closeSocketLocked();
            return false;
        }

        const int conn_res = ::connect(socket_fd_, reinterpret_cast<sockaddr*>(&server_addr_), sizeof(server_addr_));
        if (conn_res != 0 && errno != EINPROGRESS) {
            LOGW("TcpOutput: connect to ", server_ip_, ":", server_port_, " failed (errno=", errno, ")");
            tcp_connected_.store(false);
            closeSocketLocked();
            return false;
        }

        // 有界等待连接完成；该函数只在发送线程/open() 中调用，不会阻塞推理管线。
        bool connected = false;
        if (conn_res == 0) {
            connected = true;
        } else {
            pollfd pfd{};
            pfd.fd = socket_fd_;
            pfd.events = POLLOUT;
            const int timeout_ms = 500;
            int rc = ::poll(&pfd, 1, timeout_ms);
            if (rc > 0 && (pfd.revents & POLLOUT)) {
                int so_error = 0;
                socklen_t len = sizeof(so_error);
                if (getsockopt(socket_fd_, SOL_SOCKET, SO_ERROR, &so_error, &len) == 0 && so_error == 0) {
                    connected = true;
                } else {
                    LOGW("TcpOutput: connect SO_ERROR=", so_error);
                }
            } else {
                LOGW("TcpOutput: connect timeout after ", timeout_ms, "ms");
            }
        }

        if (!connected) {
            tcp_connected_.store(false);
            closeSocketLocked();
            return false;
        }

        tcp_connected_.store(true);
        has_reconnect_attempt_ = false;
        reconnect_backoff_ = reconnect_backoff_initial_;
        LOGI("TcpOutput: connected to ", server_ip_, ":", server_port_);

        if (const char* env_snd = std::getenv("RKAPP_TCP_SNDBUF")) {
            const long sz_long = std::strtol(env_snd, nullptr, 10);
            // Clamp to [1, 256 MiB] to guard against negative/overflow values
            const int sz = (sz_long > 0 && sz_long <= 256L * 1024 * 1024)
                ? static_cast<int>(sz_long) : 0;
            if (sz > 0 && setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz)) == 0) {
                LOGI("TcpOutput: SO_SNDBUF set to ", sz);
            }
        }

        return true;
}

bool TcpOutput::attemptReconnect() {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(socket_mtx_);
        if (has_reconnect_attempt_ &&
            now - last_reconnect_attempt_ < reconnect_backoff_) {
            return tcp_connected_.load();
        }

        last_reconnect_attempt_ = now;
        has_reconnect_attempt_ = true;
        if (setup_socket_locked()) {
            return true;
        }
        reconnect_backoff_ = std::min(reconnect_backoff_ * 2, reconnect_backoff_max_);
        return false;
}

bool TcpOutput::sendBuffer(QueuedPayload& payload) {
        while (payload.offset < payload.data.size()) {
            ssize_t sent = -1;
            int send_errno = 0;
            {
                std::lock_guard<std::mutex> lock(socket_mtx_);
                if (!tcp_connected_.load() || socket_fd_ < 0) {
                    return false;
                }
                sent = ::send(socket_fd_,
                              payload.data.data() + payload.offset,
                              payload.data.size() - payload.offset,
                              MSG_NOSIGNAL);
                if (sent < 0) {
                    send_errno = errno;
                }
            }
            if (sent > 0) {
                payload.offset += static_cast<size_t>(sent);
                continue;
            }

            if (sent < 0 && send_errno == EINTR) {
                continue;
            }

            if (sent < 0 && (send_errno == EAGAIN || send_errno == EWOULDBLOCK)) {
                // Brief poll before returning to avoid busy-spin when kernel send buffer is full.
                int fd_copy;
                {
                    std::lock_guard<std::mutex> lock(socket_mtx_);
                    fd_copy = socket_fd_;
                }
                if (fd_copy >= 0) {
                    pollfd pfd{};
                    pfd.fd = fd_copy;
                    pfd.events = POLLOUT;
                    ::poll(&pfd, 1, 1);  // 1ms backoff
                }
                return false;
            }

            LOGW("TcpOutput: send failed (errno=", send_errno, ")");
            closeSocket();
            return false;
        }

        return true;
}

} // namespace rkapp::output

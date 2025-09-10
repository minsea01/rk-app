#include "rkapp/output/TcpOutput.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <net/if.h>
#include <netinet/tcp.h>

namespace rkapp::output {

TcpOutput::TcpOutput() = default;
TcpOutput::~TcpOutput() { close(); }

bool TcpOutput::open(const std::string& config) {
        // Parse config string. Supported formats (comma-separated keys after first ip:port):
        //   "<ip>:<port>" 
        //   "<ip>:<port>,file:<path.jsonl>"
        //   "<ip>:<port>,iface:<ethX>"          (SO_BINDTODEVICE, may require root)
        //   "<ip>:<port>,bind_ip:<local_ip>"    (bind local source address)
        std::istringstream ss(config);
        std::string part;

        // Parse IP and port (first token)
        if (std::getline(ss, part, ',')) {
            size_t colon_pos = part.find(':');
            if (colon_pos != std::string::npos) {
                server_ip_ = part.substr(0, colon_pos);
                server_port_ = std::stoi(part.substr(colon_pos + 1));
            } else {
                std::cerr << "Invalid TCP config format. Expected ip:port, got: " << part << std::endl;
                return false;
            }
        }

        // Parse additional comma-separated options
        while (std::getline(ss, part, ',')) {
            if (part.rfind("file:", 0) == 0) {
                file_path_ = part.substr(5);
                enable_file_output_ = !file_path_.empty();
            } else if (part.rfind("iface:", 0) == 0) {
                bind_interface_ = part.substr(6);
            } else if (part.rfind("bind_ip:", 0) == 0) {
                bind_ip_ = part.substr(8);
            } else if (!part.empty()) {
                std::cerr << "TcpOutput: unknown option '" << part << "' (ignored)" << std::endl;
            }
        }

        // Create TCP socket
        socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd_ < 0) {
            std::cerr << "Failed to create TCP socket" << std::endl;
            return false;
        }

        // Enable TCP_NODELAY to reduce latency unless explicitly disabled via option later (not implemented yet)
        {
            int flag = 1;
            if (setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) != 0) {
                std::perror("TcpOutput: TCP_NODELAY failed");
            }
        }

        // If bind_ip_ provided, bind local source address before connect
        if (!bind_ip_.empty()) {
            struct sockaddr_in local{};
            local.sin_family = AF_INET;
            local.sin_port = 0; // any
            if (inet_pton(AF_INET, bind_ip_.c_str(), &local.sin_addr) <= 0) {
                std::cerr << "TcpOutput: invalid bind_ip: " << bind_ip_ << std::endl;
            } else if (bind(socket_fd_, (struct sockaddr*)&local, sizeof(local)) < 0) {
                std::perror("TcpOutput: bind(bind_ip) failed");
            } else {
                std::cout << "TcpOutput: bound local ip " << bind_ip_ << std::endl;
            }
        }

        // If bind_interface_ provided, attempt SO_BINDTODEVICE
        if (!bind_interface_.empty()) {
            if (setsockopt(socket_fd_, SOL_SOCKET, SO_BINDTODEVICE, bind_interface_.c_str(), bind_interface_.size()) != 0) {
                std::perror("TcpOutput: SO_BINDTODEVICE failed (need CAP_NET_RAW/root?)");
            } else {
                std::cout << "TcpOutput: bound to iface " << bind_interface_ << std::endl;
            }
        }

        // Setup server address
        server_addr_.sin_family = AF_INET;
        server_addr_.sin_port = htons(server_port_);
        if (inet_pton(AF_INET, server_ip_.c_str(), &server_addr_.sin_addr) <= 0) {
            std::cerr << "Invalid IP address: " << server_ip_ << std::endl;
            ::close(socket_fd_);
            socket_fd_ = -1;
            return false;
        }

        // Try to connect
        if (connect(socket_fd_, (struct sockaddr*)&server_addr_, sizeof(server_addr_)) < 0) {
            std::cerr << "Failed to connect to " << server_ip_ << ":" << server_port_
                      << " (This is normal if server is not running)" << std::endl;
            // Don't return false, allow to continue with file output only
            tcp_connected_ = false;
        } else {
            tcp_connected_ = true;
            std::cout << "Connected to TCP server " << server_ip_ << ":" << server_port_ << std::endl;
            // Optionally enlarge send buffer if env provided: RKAPP_TCP_SNDBUF (bytes)
            const char* env_snd = std::getenv("RKAPP_TCP_SNDBUF");
            if (env_snd) {
                int sz = std::atoi(env_snd);
                if (sz > 0) {
                    if (setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz)) != 0) {
                        std::perror("TcpOutput: SO_SNDBUF failed");
                    } else {
                        std::cout << "TcpOutput: SO_SNDBUF set to " << sz << std::endl;
                    }
                }
            }
        }

        // Open file output if specified
        if (enable_file_output_) {
            file_output_.open(file_path_, std::ios::app);
            if (!file_output_.is_open()) {
                std::cerr << "Failed to open output file: " << file_path_ << std::endl;
                enable_file_output_ = false;
            } else {
                std::cout << "Opened file output: " << file_path_ << std::endl;
            }
        }

        is_opened_ = tcp_connected_ || enable_file_output_;
        return is_opened_;
}
    
bool TcpOutput::send(const FrameResult& result) {
        if (!is_opened_) {
            return false;
        }
        
        // Create JSON string
        std::ostringstream json;
        json << "{";
        json << "\"frame_id\":" << result.frame_id << ",";
        json << "\"timestamp\":" << result.timestamp << ",";
        json << "\"width\":" << result.width << ",";
        json << "\"height\":" << result.height << ",";
        json << "\"source_uri\":\"" << result.source_uri << "\",";
        json << "\"detections\":[";
        
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& det = result.detections[i];
            if (i > 0) json << ",";
            json << "{";
            json << "\"x\":" << det.x << ",";
            json << "\"y\":" << det.y << ",";
            json << "\"w\":" << det.w << ",";
            json << "\"h\":" << det.h << ",";
            json << "\"confidence\":" << det.confidence << ",";
            json << "\"class_id\":" << det.class_id << ",";
            json << "\"class_name\":\"" << det.class_name << "\"";
            json << "}";
        }
        
        json << "]}" << std::endl;
        
        std::string json_str = json.str();
        bool success = false;
        
        // Send via TCP if connected
        if (tcp_connected_) {
            ssize_t bytes_sent = ::send(socket_fd_, json_str.c_str(), json_str.length(), 0);
            if (bytes_sent > 0) {
                success = true;
            } else {
                std::cerr << "Failed to send TCP data" << std::endl;
                tcp_connected_ = false;
            }
        }
        
        // Write to file if enabled
        if (enable_file_output_ && file_output_.is_open()) {
            file_output_ << json_str;
            file_output_.flush();
            success = true;
        }
        
        return success;
}
    
void TcpOutput::close() {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        
        if (file_output_.is_open()) {
            file_output_.close();
        }
        
        tcp_connected_ = false;
        is_opened_ = false;
}
    
bool TcpOutput::isOpened() const { return is_opened_; }
    
OutputType TcpOutput::getType() const { return OutputType::TCP; }

} // namespace rkapp::output

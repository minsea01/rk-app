#!/bin/bash
# RTP H.264 视频流发送脚本
# 适用于测试和演示

set -e

# 默认参数
HOST="127.0.0.1"
PORT="5000"
WIDTH="1280"
HEIGHT="720"
FRAMERATE="60"
BITRATE="4000"
PATTERN="smpte"

# 帮助信息
show_help() {
    cat << EOF
RTP H.264 视频流发送器

用法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -H, --host HOST     目标主机地址 (默认: $HOST)
    -p, --port PORT     目标端口 (默认: $PORT)
    -w, --width WIDTH   视频宽度 (默认: $WIDTH)
    -h, --height HEIGHT 视频高度 (默认: $HEIGHT)
    -f, --fps FPS       帧率 (默认: $FRAMERATE)
    -b, --bitrate RATE  码率(kbps) (默认: $BITRATE)
    -t, --pattern TYPE  测试图案 (默认: $PATTERN)
                        可选: smpte, snow, black, white, red, green, blue, checkers-1

示例:
    $0                                    # 使用默认参数
    $0 -H 192.168.1.100 -p 5001         # 发送到指定IP和端口
    $0 -w 1920 -h 1080 -f 30 -b 8000    # 1080p 30fps 8Mbps
    $0 -t snow                           # 雪花噪声图案

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        -f|--fps)
            FRAMERATE="$2"
            shift 2
            ;;
        -b|--bitrate)
            BITRATE="$2"
            shift 2
            ;;
        -t|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查GStreamer是否可用
if ! command -v gst-launch-1.0 &> /dev/null; then
    echo "错误: 未找到 gst-launch-1.0"
    echo "请安装 GStreamer: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
    exit 1
fi

# 显示配置信息
echo "=========================================="
echo "RTP H.264 视频流发送器"
echo "=========================================="
echo "目标地址: $HOST:$PORT"
echo "分辨率:   ${WIDTH}x${HEIGHT}"
echo "帧率:     ${FRAMERATE} fps"
echo "码率:     ${BITRATE} kbps"
echo "图案:     $PATTERN"
echo "=========================================="
echo "按 Ctrl+C 停止发送"
echo ""

# 构建GStreamer命令
GST_CMD="gst-launch-1.0 -v \
    videotestsrc is-live=true pattern=$PATTERN ! \
    video/x-raw,framerate=$FRAMERATE/1,width=$WIDTH,height=$HEIGHT ! \
    x264enc tune=zerolatency bitrate=$BITRATE speed-preset=ultrafast key-int-max=$FRAMERATE ! \
    rtph264pay pt=96 config-interval=1 ! \
    udpsink host=$HOST port=$PORT"

# 显示完整命令（用于调试）
echo "执行命令:"
echo "$GST_CMD"
echo ""

# 执行命令
exec $GST_CMD

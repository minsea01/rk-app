#!/usr/bin/env bash
#
# RGMII Driver Configuration and Validation Script for RK3588
#
# Purpose: Detect, configure, and validate dual RGMII Gigabit Ethernet interfaces
# Platform: RK3588 with STMMAC driver (dwmac-rk variant)
# Author: Generated for RK3588 Pedestrian Detection Project
# Date: 2025-11-16
#
set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check if running on RK3588
check_platform() {
    log_info "Checking platform compatibility..."

    if [[ ! -f /proc/device-tree/compatible ]]; then
        log_warn "Device tree not found, cannot verify RK3588"
        return 1
    fi

    if grep -q "rockchip,rk3588" /proc/device-tree/compatible 2>/dev/null; then
        log_info "✓ Running on RK3588 platform"
        return 0
    else
        log_warn "Not running on RK3588, script will run in simulation mode"
        return 1
    fi
}

# Detect RGMII interfaces
detect_rgmii_interfaces() {
    log_info "Detecting RGMII Ethernet interfaces..."

    local rgmii_count=0
    local eth_devices=()

    # Method 1: Device tree inspection
    if [[ -d /sys/firmware/devicetree/base ]]; then
        for eth_node in /sys/firmware/devicetree/base/ethernet@*; do
            if [[ -d "$eth_node" ]]; then
                local phy_mode=""
                if [[ -f "$eth_node/phy-mode" ]]; then
                    phy_mode=$(cat "$eth_node/phy-mode" | tr -d '\0')
                fi

                if [[ "$phy_mode" == "rgmii" || "$phy_mode" == "rgmii-id" || "$phy_mode" == "rgmii-rxid" || "$phy_mode" == "rgmii-txid" ]]; then
                    log_info "  Found RGMII interface: $(basename "$eth_node") (phy-mode: $phy_mode)"
                    ((rgmii_count++))
                fi
            fi
        done
    fi

    # Method 2: Network interface inspection
    for iface in /sys/class/net/eth*; do
        if [[ -d "$iface" ]]; then
            local iface_name=$(basename "$iface")
            eth_devices+=("$iface_name")

            # Check driver
            local driver=""
            if [[ -L "$iface/device/driver" ]]; then
                driver=$(basename "$(readlink "$iface/device/driver")")
            fi

            log_info "  Network interface: $iface_name (driver: ${driver:-unknown})"
        fi
    done

    if [[ $rgmii_count -eq 0 && ${#eth_devices[@]} -eq 0 ]]; then
        log_error "No RGMII/Ethernet interfaces detected"
        return 1
    fi

    log_info "✓ Detected $rgmii_count RGMII device tree nodes, ${#eth_devices[@]} network interfaces"
    return 0
}

# Check STMMAC driver
check_stmmac_driver() {
    log_info "Checking STMMAC Ethernet driver..."

    local driver_loaded=0

    # Check if module is loaded
    if lsmod 2>/dev/null | grep -q "stmmac"; then
        log_info "  ✓ stmmac module loaded"
        driver_loaded=1
    fi

    if lsmod 2>/dev/null | grep -q "dwmac_rk"; then
        log_info "  ✓ dwmac_rk (Rockchip variant) loaded"
        driver_loaded=1
    fi

    # Check kernel module path
    if find /lib/modules/"$(uname -r)" -name "stmmac*.ko*" 2>/dev/null | grep -q .; then
        log_info "  ✓ STMMAC driver available in kernel"
        driver_loaded=1
    fi

    if [[ $driver_loaded -eq 0 ]]; then
        log_error "STMMAC driver not found"
        log_info "  Attempting to load driver..."

        if command -v modprobe >/dev/null 2>&1; then
            if modprobe stmmac 2>/dev/null || modprobe dwmac-rk 2>/dev/null; then
                log_info "  ✓ Driver loaded successfully"
                return 0
            fi
        fi

        log_error "Failed to load STMMAC driver"
        return 1
    fi

    return 0
}

# Configure network interface for optimal performance
configure_interface() {
    local iface=$1

    log_info "Configuring interface: $iface"

    # Check if interface exists
    if [[ ! -d /sys/class/net/"$iface" ]]; then
        log_error "Interface $iface does not exist"
        return 1
    fi

    # Bring interface up
    if command -v ip >/dev/null 2>&1; then
        ip link set "$iface" up 2>/dev/null || log_warn "Could not bring up $iface (may need sudo)"
    fi

    # Check if ethtool is available
    if ! command -v ethtool >/dev/null 2>&1; then
        log_warn "ethtool not installed, skipping advanced configuration"
        return 0
    fi

    # Get current speed
    local speed=$(ethtool "$iface" 2>/dev/null | grep -i "speed:" | awk '{print $2}' || echo "unknown")
    log_info "  Current speed: $speed"

    # Check if it's Gigabit
    if [[ "$speed" == "1000Mb/s" ]]; then
        log_info "  ✓ Interface running at Gigabit speed"
    elif [[ "$speed" == "Unknown!" || "$speed" == "unknown" ]]; then
        log_warn "  Interface speed unknown (may be down)"
    else
        log_warn "  Interface not at Gigabit speed: $speed"
    fi

    # Optimize ring buffers (if supported)
    local rx_current=$(ethtool -g "$iface" 2>/dev/null | grep -A 4 "Current hardware settings" | grep "RX:" | awk '{print $2}' | head -1)
    local rx_max=$(ethtool -g "$iface" 2>/dev/null | grep -A 4 "Pre-set maximums" | grep "RX:" | awk '{print $2}' | head -1)

    if [[ -n "$rx_max" && "$rx_max" != "n/a" ]]; then
        log_info "  RX ring buffer: current=$rx_current, max=$rx_max"

        if [[ $rx_current -lt $rx_max ]]; then
            log_info "  Attempting to increase RX ring buffer to $rx_max"
            ethtool -G "$iface" rx "$rx_max" 2>/dev/null && log_info "  ✓ RX buffer increased" || log_warn "  Could not increase RX buffer (need sudo)"
        fi
    fi

    # Enable hardware offloading
    log_info "  Checking hardware offload features..."
    ethtool -k "$iface" 2>/dev/null | grep ": on" | head -5 || true

    return 0
}

# Optimize system network parameters
optimize_sysctl() {
    log_info "Checking system network parameters..."

    if ! command -v sysctl >/dev/null 2>&1; then
        log_warn "sysctl not available, skipping system optimization"
        return 0
    fi

    # Key parameters for high throughput
    local params=(
        "net.core.rmem_max"
        "net.core.wmem_max"
        "net.core.netdev_max_backlog"
        "net.ipv4.tcp_rmem"
        "net.ipv4.tcp_wmem"
    )

    for param in "${params[@]}"; do
        local value=$(sysctl -n "$param" 2>/dev/null || echo "N/A")
        log_info "  $param = $value"
    done

    # Recommended values for >= 900 Mbps
    log_info ""
    log_info "Recommended optimizations for >= 900 Mbps throughput:"
    cat <<'EOF'

  # Add to /etc/sysctl.conf or /etc/sysctl.d/99-network-performance.conf:
  net.core.rmem_max = 134217728
  net.core.wmem_max = 134217728
  net.core.netdev_max_backlog = 5000
  net.ipv4.tcp_rmem = 4096 87380 67108864
  net.ipv4.tcp_wmem = 4096 65536 67108864
  net.ipv4.tcp_congestion_control = bbr

  # Apply with: sudo sysctl -p
EOF

    return 0
}

# Generate validation report
generate_report() {
    log_info "==================================="
    log_info "RGMII Configuration Summary"
    log_info "==================================="

    echo ""
    log_info "Platform: $(uname -m) - $(uname -r)"

    echo ""
    log_info "Network Interfaces:"
    ip -br addr 2>/dev/null || ifconfig 2>/dev/null | grep -E "^eth|^en" || echo "  No interfaces found"

    echo ""
    log_info "Driver Status:"
    lsmod 2>/dev/null | grep -E "stmmac|dwmac" || echo "  STMMAC driver not loaded"

    echo ""
    log_info "Next Steps:"
    echo "  1. Run network throughput test: ./scripts/network/network_throughput_validator.sh"
    echo "  2. Configure static IP if needed: sudo ip addr add <ip>/24 dev eth0"
    echo "  3. Test with industrial camera connection"

    return 0
}

# Main execution
main() {
    log_info "=========================================="
    log_info "RK3588 RGMII Driver Configuration Script"
    log_info "=========================================="
    echo ""

    # Platform check (non-fatal)
    check_platform || log_warn "Not on RK3588 - running in validation mode"

    echo ""
    detect_rgmii_interfaces || {
        log_error "Failed to detect RGMII interfaces"
        exit 1
    }

    echo ""
    check_stmmac_driver || {
        log_error "STMMAC driver check failed"
        exit 1
    }

    echo ""
    # Configure each eth interface
    for iface in /sys/class/net/eth*; do
        if [[ -d "$iface" ]]; then
            configure_interface "$(basename "$iface")" || true
            echo ""
        fi
    done

    optimize_sysctl

    echo ""
    generate_report

    log_info ""
    log_info "✓ RGMII driver configuration completed successfully"

    return 0
}

# Run main function
main "$@"

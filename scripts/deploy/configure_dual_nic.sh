#!/bin/bash
# Configure dual Gigabit Ethernet (RGMII) on RK3588
# Port 1: Camera feed input (1080P stream)
# Port 2: Detection results output (TCP/UDP uplink)
#
# Usage:
#   sudo ./configure_dual_nic.sh
#   sudo ./configure_dual_nic.sh --interface eth0 eth1
#
# Requirements:
#   - Netplan or NetworkManager configured
#   - Kernel ≥5.10 with RK3588 RGMII drivers

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Default interfaces (RK3588 typically has eth0, eth1)
ETH0="${1:-eth0}"
ETH1="${2:-eth1}"

log_info "Configuring dual NIC: $ETH0 (camera input) + $ETH1 (detection output)"

# 1. Check current network setup
log_info "Checking current network configuration..."
ip link show || log_error "Failed to get network interfaces"

# 2. Get interface status
ETH0_STATUS=$(ip link show $ETH0 | grep "state UP" | wc -l)
ETH1_STATUS=$(ip link show $ETH1 | grep "state UP" | wc -l)

log_info "Interface $ETH0: $([ $ETH0_STATUS -eq 1 ] && echo 'UP' || echo 'DOWN')"
log_info "Interface $ETH1: $([ $ETH1_STATUS -eq 1 ] && echo 'UP' || echo 'DOWN')"

# 3. Bring up interfaces with static IPs
log_info "Configuring IP addresses..."

# Port 1 (Camera input): 192.168.1.100
ip addr add 192.168.1.100/24 dev $ETH0 2>/dev/null || log_warn "Could not add IP to $ETH0 (may already exist)"
ip link set $ETH0 up

# Port 2 (Detection output): 192.168.2.100
ip addr add 192.168.2.100/24 dev $ETH1 2>/dev/null || log_warn "Could not add IP to $ETH1 (may already exist)"
ip link set $ETH1 up

log_info "IP Configuration:"
echo "  $ETH0 (Camera Input):     192.168.1.100/24"
echo "  $ETH1 (Detection Output): 192.168.2.100/24"

# 4. Enable IP forwarding for routing
log_info "Enabling IP forwarding..."
sysctl -w net.ipv4.ip_forward=1 >/dev/null

# 5. Verify connectivity
log_info "Verifying network interfaces..."
ip addr show | grep -E "inet|eth"

# 6. Test interface speed and status
log_info "Network speed and status:"
for iface in $ETH0 $ETH1; do
    if command -v ethtool &> /dev/null; then
        SPEED=$(ethtool $iface 2>/dev/null | grep "Speed" || echo "Unknown")
        log_info "  $iface: $SPEED"
    fi
done

# 7. Persist configuration (Netplan)
log_info "Persisting configuration with Netplan..."

# Create Netplan configuration
cat > /etc/netplan/02-rk3588-dual-nic.yaml <<EOF
# Dual NIC configuration for RK3588
# Port 1 (eth0): Camera input (192.168.1.0/24)
# Port 2 (eth1): Detection output (192.168.2.0/24)

network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
      mtu: 1500
      routes:
        - to: 192.168.1.0/24
          via: 192.168.1.1
          metric: 100
      description: "Camera Input (1080P stream)"

    eth1:
      dhcp4: no
      addresses:
        - 192.168.2.100/24
      gateway4: 192.168.2.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
      mtu: 1500
      routes:
        - to: 192.168.2.0/24
          via: 192.168.2.1
          metric: 101
      description: "Detection Output (results upload)"
EOF

chmod 600 /etc/netplan/02-rk3588-dual-nic.yaml
log_info "Netplan config saved: /etc/netplan/02-rk3588-dual-nic.yaml"

# 8. Apply Netplan configuration
log_info "Applying Netplan configuration..."
netplan apply 2>/dev/null || log_warn "Netplan apply may require reboot to take full effect"

# 9. Test throughput (if iperf3 available)
log_info "Throughput baseline (requires iperf3 server on remote):"
echo "  To test bandwidth: iperf3 -c <server_ip> -p 5201"
echo "  Expected: ≥900 Mbps on each port"

# 10. Display final configuration
log_info "Final network configuration:"
ip addr show | grep -A2 "inet "
log_info "Routing table:"
ip route show

# 11. Summary
log_info "=========================================="
log_info "Dual NIC configuration complete!"
log_info "=========================================="
echo ""
echo "Network Layout:"
echo "  ┌─── RK3588 ───────────────────┐"
echo "  │  eth0: 192.168.1.100        │  ← Camera feed input"
echo "  │  eth1: 192.168.2.100        │  ← Detection output"
echo "  └──────────────────────────────┘"
echo ""
echo "Next steps:"
echo "  1. Verify connectivity:"
echo "     ping -c 5 192.168.1.1  # Should connect to camera network"
echo "     ping -c 5 192.168.2.1  # Should connect to server network"
echo ""
echo "  2. Test throughput (when server available):"
echo "     iperf3 -c <camera_ip> -B 192.168.1.100 -t 10"
echo "     iperf3 -c <server_ip> -B 192.168.2.100 -t 10"
echo ""
echo "  3. Start detection with dual output:"
echo "     ./scripts/deploy/rk3588_run.sh --config config/dual_nic.yaml"
echo ""
echo "⚠️  To persist across reboots, run: netplan apply"
echo "⚠️  To revert config, run: rm /etc/netplan/02-rk3588-dual-nic.yaml && netplan apply"

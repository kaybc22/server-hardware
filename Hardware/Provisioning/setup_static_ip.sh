#!/bin/bash
# =============================================================================
# Automated Static IP Configuration - Improved DNS Detection
# =============================================================================

echo "=== Static IP Configuration Tool ==="
echo "This script will create a static IP config using Netplan"
echo "-----------------------------------------------------"

# Step 1: Detect interface, IP, Gateway
INTERFACE=$(ip -o link show | awk -F': ' '/state UP/ {print $2}' | grep -E '^(en|eth)' | head -n1)

if [ -z "$INTERFACE" ]; then
    echo "No active Ethernet interface detected."
    exit 1
fi

CURRENT_IP=$(ip -4 addr show "$INTERFACE" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}(?=/)' | head -n1)
CURRENT_PREFIX=$(ip -4 addr show "$INTERFACE" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}/\K\d+')
CURRENT_GW=$(ip route | grep default | grep "$INTERFACE" | awk '{print $3}' | head -n1)

echo "Active Interface : $INTERFACE"
echo "Current IP       : $CURRENT_IP/$CURRENT_PREFIX"
echo "Current Gateway  : $CURRENT_GW"

# Step 2: Improved DNS Detection
echo -e "\nDetecting DNS servers..."

# Try resolvectl first (modern systemd)
if command -v resolvectl >/dev/null 2>&1; then
    DNS_SERVERS=$(resolvectl status "$INTERFACE" 2>/dev/null | grep -oP 'DNS Servers: \K.*' | head -1)
    [[ -z "$DNS_SERVERS" ]] && DNS_SERVERS=$(resolvectl status | grep -oP 'DNS Servers: \K.*' | head -1)
fi

# Fallback to /etc/resolv.conf (filter out localhost)
if [[ -z "$DNS_SERVERS" ]]; then
    DNS_SERVERS=$(grep '^nameserver' /etc/resolv.conf 2>/dev/null | awk '{print $2}' | grep -v '^127\.' | tr '\n' ' ')
fi

echo "Detected DNS       : $DNS_SERVERS"

# Step 3: Ask user whether to use current values
read -p "Do you want to use the current IP, Gateway, and DNS as static configuration? (y/N): " USE_CURRENT

if [[ "$USE_CURRENT" =~ ^[Yy]$ ]]; then
    IP_ADDRESS="$CURRENT_IP/$CURRENT_PREFIX"
    GATEWAY="$CURRENT_GW"
    echo "✅ Using current network settings as static IP."
else
    echo -e "\nPlease enter static IP details manually:"
    read -p "IP Address with prefix (e.g. 172.31.34.151/16): " IP_ADDRESS
    read -p "Gateway (e.g. 172.31.0.1): " GATEWAY
    read -p "DNS servers (space separated): " DNS_SERVERS
fi

# Step 4: Create Netplan config
CONFIG_FILE="/etc/netplan/50-static-ip.yaml"

cat <<EOF | sudo tee $CONFIG_FILE > /dev/null
network:
  version: 2
  renderer: networkd
  ethernets:
    $INTERFACE:
      dhcp4: no
      addresses: [$IP_ADDRESS]
      routes:
        - to: default
          via: $GATEWAY
      nameservers:
        addresses: [$(echo $DNS_SERVERS | sed 's/ /, /g')]
EOF

echo -e "\n✅ Static IP configuration file created successfully!"
echo "File location: $CONFIG_FILE"
echo -e "\nPreview:"
cat "$CONFIG_FILE"

# Step 5: Apply configuration
read -p "Apply this configuration now? (y/N): " CONFIRM

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Applying Netplan..."
    sudo netplan generate
    if sudo netplan apply; then
        echo "✅ Static IP applied successfully!"
        echo "New IP → $IP_ADDRESS"
    else
        echo "❌ Failed to apply. Please check the config file."
    fi
else
    echo "Configuration saved but not applied."
fi

echo -e "\nDone!"

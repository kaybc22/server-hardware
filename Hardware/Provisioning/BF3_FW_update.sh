#!/bin/bash
# BlueField Firmware Update Script
# Author: Khoa
# Description: Automates BFB firmware update across multiple rshim devices

set -euo pipefail

#BFB_URL="https://content.mellanox.com/BlueField/BFBs/Ubuntu22.04/bf-bundle-3.0.0-135_25.04_ubuntu-22.04_prod.bfb"

#https://content.mellanox.com/BlueField/FW-Bundle/bf-fwbundle-3.1.0-82_25.07-prod.bfb
BFB_URL="https://content.mellanox.com/BlueField/FW-Bundle/bf-fwbundle-3.1.0-82_25.07-prod.bfb"
BFB_FILE="bf-bundle-3.0.0-135_25.04_ubuntu-22.04_prod.bfb"
CFG_FILE="bf.cfg"
MMC_DEV="/dev/mmcblk0"

echo "=== BlueField Firmware Update Started ==="

# 1. Restart rshim service
#echo "[INFO] Restarting rshim service..."
#systemctl stop rshim || true
#systemctl start rshim

# 2. Download BFB if not already present
if [ ! -f "$BFB_FILE" ]; then
    echo "[INFO] Downloading BFB: $BFB_URL"
    wget -q "$BFB_URL" -O "$BFB_FILE"
else
    echo "[INFO] Using existing BFB file: $BFB_FILE"
fi

# 3. Create config file
echo "[INFO] Creating config file: $CFG_FILE"
cat > "$CFG_FILE" <<EOF
device=$MMC_DEV
WITH_NIC_FW_UPDATE=yes
EOF

# 4. Run update on specific rshim devices
update_specific() {
DEVICES="0 1 2 5 7 8 9"
for i in $DEVICES; do
    if [ -e "/dev/rshim$i" ]; then
        echo "[INFO] Updating rshim$i ..."
        bfb-install --bfb "$BFB_FILE" -c "$CFG_FILE" --rshim "rshim$i"
    else
        echo "[WARN] /dev/rshim$i not found, skipping..."
    fi
done
}

# 5. Run update on all detected rshim devices
echo "[INFO] Running update on all detected rshim devices..."
update_all() {
for dev in /dev/rshim*; do
    [ -e "$dev" ] || continue
    echo "[INFO] Updating $dev ..."
    bfb-install --bfb "$BFB_FILE" -c "$CFG_FILE" --rshim "$(basename "$dev")"
done
}

#call update all fuction
#update_all

#update_specific

echo "=== Firmware Update Completed ==="

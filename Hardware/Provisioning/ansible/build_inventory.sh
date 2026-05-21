#!/bin/bash
set -e

echo "=== Build Ansible Inventory from IP List ==="

read -p "Enter IP list file (e.g., nodes.txt): " FILE

if [ ! -f "$FILE" ]; then
    echo "File not found."
    exit 1
fi

# Read IPs into array
mapfile -t IPS < "$FILE"

if [ ${#IPS[@]} -eq 0 ]; then
    echo "No IPs found in file."
    exit 1
fi

MASTER_IP=${IPS[0]}
WORKER_IPS=("${IPS[@]:1}")

OUTFILE="inventory.yaml"

echo "Generating $OUTFILE..."

{
echo "all:"
echo "  hosts:"
echo "    master:"
echo "      ansible_host: $MASTER_IP"
echo ""

i=1
for ip in "${WORKER_IPS[@]}"; do
    echo "    dgx_node_$i:"
    echo "      ansible_host: $ip"
    echo ""
    ((i++))
done

} > "$OUTFILE"

echo "Inventory created: $OUTFILE"


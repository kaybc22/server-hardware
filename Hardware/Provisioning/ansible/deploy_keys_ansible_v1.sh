#!/bin/bash
set -e

echo "=== SSH Key Deployment + Ansible Install Tool (root mode) ==="

# 1. Ensure SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "No SSH key found. Generating one..."
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

# 2. Install sshpass if missing
if ! command -v sshpass >/dev/null; then
    echo "Installing sshpass..."
    sudo apt update && sudo apt install -y sshpass
fi

# 3. Ask for root password (same for all nodes)
read -s -p "Enter root password for all nodes: " ROOTPASS
echo ""

# 4. Ask for IP list file
read -p "Enter IP list file (e.g., ansible_node_ip.txt): " FILE

if [ ! -f "$FILE" ]; then
    echo "File not found."
    exit 1
fi

mapfile -t IP_LIST < "$FILE"

# 5. Deploy SSH key to each node (root)
for ip in "${IP_LIST[@]}"; do
    echo "Deploying SSH key to root@$ip..."
    sshpass -p "$ROOTPASS" ssh-copy-id -o StrictHostKeyChecking=no root@"$ip"
done

echo "=== SSH key deployment completed ==="

# 6. Install Ansible on each node (root)
echo "=== Installing Ansible on all nodes ==="

for ip in "${IP_LIST[@]}"; do
    echo "Installing Ansible on $ip..."
    ssh root@"$ip" "apt update &&
                    apt install -y software-properties-common &&
                    add-apt-repository --yes --update ppa:ansible/ansible &&
                    apt install -y ansible"
done

echo "=== All nodes have Ansible installed ==="


#!/bin/bash
set -e

echo "=== SSH Key Deployment Tool ==="

# 1. Ensure SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "No SSH key found. Generating one..."
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

# 2. Install dependencies
if ! command -v sshpass >/dev/null; then
    echo "Installing sshpass..."
    sudo apt update && sudo apt install -y sshpass
fi

# 3. Ask user for input method
echo "Choose input method:"
echo "1) Use IP list from file"
echo "2) Enter a single IP manually"
echo "3) Use default IP range (172.31.36.195–198)"
read -p "Select option (1/2/3): " choice

IP_LIST=()

case "$choice" in
    1)
        read -p "Enter path to IP list file: " file
        if [ ! -f "$file" ]; then
            echo "File not found."
            exit 1
        fi
        mapfile -t IP_LIST < "$file"
        ;;
    2)
        read -p "Enter the IP address: " ip
        IP_LIST+=("$ip")
        ;;
    3)
        IP_LIST=(172.31.36.{195..198})
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# 4. Deploy SSH key to each node
for ip in "${IP_LIST[@]}"; do
    echo "Deploying SSH key to $ip..."
    sshpass -p Super123 ssh-copy-id -o StrictHostKeyChecking=no root@"$ip"
done

echo "=== SSH key deployment completed ==="


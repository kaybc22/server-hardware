#!/bin/bash
# copy_ssh_keys.sh — automate copying SSH public key to cluster nodes
# Usage:
#   ./copy_ssh_keys.sh ip1 ip2 ip3
#   ./copy_ssh_keys.sh -f ip_list.txt
#   ./copy_ssh_keys.sh -u root -p "yourpassword" 192.168.1.101 192.168.1.102 192.168.1.103
#   ssh root@192.168.1.101 hostname
# Notes:
#   - Requires sshpass
#   - Use a valid SSH password for all nodes (same user account)
#   - Will append key to ~/.ssh/authorized_keys on each node

PUBKEY="${HOME}/.ssh/id_ed25519.pub"
USER="${USER}"
PASS=""
IP_LIST=()

usage() {
  echo "Usage:"
  echo "  $0 ip1 ip2 ip3"
  echo "  $0 -f ip_list.txt"
  echo "Options:"
  echo "  -u USERNAME   SSH user (default: current user)"
  echo "  -p PASSWORD   SSH password (prompt if omitted)"
  echo "  -f FILE       File with one IP per line"
  exit 1
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u) USER="$2"; shift 2 ;;
    -p) PASS="$2"; shift 2 ;;
    -f) 
      if [[ -f "$2" ]]; then
        mapfile -t IP_LIST < "$2"
      else
        echo "File not found: $2"
        exit 1
      fi
      shift 2 ;;
    -h|--help) usage ;;
    *) IP_LIST+=("$1"); shift ;;
  esac
done

if [[ ${#IP_LIST[@]} -eq 0 ]]; then
  usage
fi

if [[ ! -f "$PUBKEY" ]]; then
  echo "Error: Public key not found at $PUBKEY"
  echo "Run 'ssh-keygen -t ed25519' first."
  exit 1
fi

# Prompt for password if not provided
if [[ -z "$PASS" ]]; then
  read -s -p "Enter SSH password for user $USER: " PASS
  echo ""
fi

echo "Using public key: $PUBKEY"
echo "Copying to nodes: ${IP_LIST[*]}"
echo ""

# --- Main Loop ---
for HOST in "${IP_LIST[@]}"; do
  echo ">>> Copying key to $HOST ..."
  sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$USER@$HOST" "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  sshpass -p "$PASS" scp -o StrictHostKeyChecking=no "$PUBKEY" "$USER@$HOST:~/.ssh/tmpkey.pub"
  sshpass -p "$PASS" ssh "$USER@$HOST" "cat ~/.ssh/tmpkey.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && rm ~/.ssh/tmpkey.pub"
  
  # Verify passwordless access
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$USER@$HOST" 'echo SSH OK' &>/dev/null; then
    echo "? Passwordless SSH confirmed for $HOST"
  else
    echo "??  Failed to verify passwordless SSH for $HOST"
  fi
  echo ""
done

echo "All done."

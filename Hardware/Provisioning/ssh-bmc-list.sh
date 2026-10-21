#!/bin/bash
# ssh-bmc-list.sh — SSH each "IP MAC" line in a text file
# Usage: ./ssh-bmc-list.sh hosts.txt [remote command...]
# ./ssh-bmc-list.sh hosts.txt "hostname; ip -br addr; ipmitool lan print 1"
# Default remote command prints BMC LAN IP and MAC.

set -uuo pipefail

FILE="${1:-}"
shift || true
REMOTE_CMD="${*:-ipmitool lan print 1 | egrep -i 'IP Address|MAC Address'}"

USER="service"
PASS="0penBmc"

if [[ -z "$FILE" || ! -f "$FILE" ]]; then
  echo "Usage: $0 <ip-mac-file> [remote command...]" >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass is required: apt install sshpass  or  dnf install sshpass" >&2
  exit 1
fi

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=8
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o LogLevel=ERROR
)

while read -r ip mac rest; do
  [[ -z "${ip:-}" || "$ip" =~ ^# ]] && continue
  echo "===== $ip  file_mac=${mac:-n/a} ====="
  if ! sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "${USER}@${ip}" "$REMOTE_CMD"; then
    echo "FAILED $ip" >&2
  fi
  echo
done < "$FILE"

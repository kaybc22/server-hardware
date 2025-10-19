#!/bin/bash
# NIC.sh - 
# Usage:
#   ./NIC.sh loopback
#   ./NIC.sh remote <REMOTE_IP> <DEV1> <DEV2>

set -euo pipefail

MODE=${1:-}
REMOTE_IP=${2:-}
DEV1=${3:-mlx5_6}
DEV2=${4:-mlx5_7}

PORT=18518
SIZE=1048576        # 1MB
DURATION=2400       # 40 minutes
IB_PORT=1
QP=1
POST_LIST=1
COMMON_FLAGS="--port=$PORT --size=$SIZE --duration=$DURATION --bidirectional --connection=RC --qp=$QP --report_gbits --CPU-freq --post_list=$POST_LIST"

log_file="ib_test_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "$MODE" ]]; then
    echo "Usage:"
    echo "  Loopback: $0 loopback"
    echo "  Remote:   $0 remote <REMOTE_IP> <DEV1> <DEV2>"
    exit 1
fi

echo "=========================================="
echo "?? Mode:          $MODE"
echo "?? Device Pair:   $DEV1 <-> $DEV2"
[[ "$MODE" == "remote" ]] && echo "?? Remote Target:  $REMOTE_IP"
echo "?? Log File:      $log_file"
echo "=========================================="
echo

# Function to run ib_write_bw
run_ib_test() {
    local dev=$1
    local extra_arg=$2
    echo "?? Running ib_write_bw on $dev ..."
    echo "------------------------------------------" | tee -a "$log_file"
    echo "[Device: $dev]" | tee -a "$log_file"
    (time ib_write_bw --ib-dev="$dev" --ib-port="$IB_PORT" $COMMON_FLAGS $extra_arg 2>&1) | tee -a "$log_file"
    echo | tee -a "$log_file"
}

if [[ "$MODE" == "loopback" ]]; then
    echo "?? Running LOOPBACK test between $DEV1 and $DEV2"
    echo

    # Start server side (DEV1)
    ib_write_bw --ib-dev="$DEV1" --ib-port="$IB_PORT" $COMMON_FLAGS &
    SERVER_PID=$!
    sleep 3

    # Run client side (DEV2)
    run_ib_test "$DEV2" "127.0.0.1"

    # Clean up
    kill $SERVER_PID 2>/dev/null || true

#SSH key-based authentication
elif [[ "$MODE" == "remote" ]]; then
    if [[ -z "$REMOTE_IP" ]]; then
        echo "? Remote mode requires IP argument!"
        echo "Example: $0 remote 192.168.1.100 mlx5_6 mlx5_7"
        exit 1
    fi

    echo "?? Running REMOTE test between local:$DEV1 and remote:$DEV2 ($REMOTE_IP)"
    echo

    # Start ib_write_bw on remote host via SSH
    echo "?? Starting remote ib_write_bw server on $REMOTE_IP ..."
    ssh -n "$REMOTE_IP" "nohup ib_write_bw --ib-dev=$DEV2 --ib-port=$IB_PORT $COMMON_FLAGS > /tmp/ib_server.log 2>&1 &"

    sleep 5  # Give it time to start

    # Run client on local host
    run_ib_test "$DEV1" "$REMOTE_IP"

    # Optionally fetch the remote log
    echo "?? Fetching remote log from $REMOTE_IP ..."
    scp "$REMOTE_IP:/tmp/ib_server.log" "./remote_ib_server_$(date +%s).log" >/dev/null 2>&1 || true

    echo "?? Cleaning up remote process ..."
    ssh "$REMOTE_IP" "pkill -f ib_write_bw" >/dev/null 2>&1 || true

else
    echo "? Invalid mode: $MODE (must be 'loopback' or 'remote')"
    exit 1
fi

echo "? Test complete. Results saved to $log_file"


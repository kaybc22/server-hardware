#!/bin/bash

# List of mlx5 parent device numbers
#DEVICES=(0 1 6 11 4 5 14 15)  #B300
#devices=(0d 35 8d b6) (6e 5b ec da) (23:00.0 -- 23:00.2) #Rubin
devices=(0 6 12 14)

CMD="/opt/mellanox/iproute2/sbin/rdma"

for dev in "${devices[@]}"; do
    echo "Adding smi${dev}..."

    sudo $CMD dev add smi${dev} type SMI parent mlx5_${dev}

    guid=$(ibstat smi${dev} 2>/dev/null | awk '/Port GUID:/ {print $3; exit}')

    if [ -n "$guid" ]; then
        echo "Starting opensm for smi${dev} (GUID: $guid)"
        opensm -g "$guid" >/tmp/opensm_${dev}.log 2>&1 &
    else
        echo "Failed to get GUID for smi${dev}"
    fi
done
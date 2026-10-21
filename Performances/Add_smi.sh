#!/bin/bash

# List of mlx5 parent device numbers
#DEVICES=(0 1 6 11 4 5 14 15)  #B300
#devices=(0 6 10 12 14 16 20 22) #Rubin
devices=(6 10 14 16)

# Base command path
CMD="/opt/mellanox/iproute2/sbin/rdma"

for dev in "${devices[@]}"; do
    echo "Adding smi-mlx5_${dev}..."
    sudo $CMD dev add smi${dev} type SMI parent mlx5_${dev}
done

#for i in $(ibstat smi${dev} | grep -i "Port GUID:" | head -1 | awk '{print $3}'); do echo $i opensm -g $i & ; done

echo "All RDMA SMI devices added."

#v2
#for dev in "${devices[@]}"; do
#    echo "Adding smi${dev}..."

#   sudo $CMD dev add smi${dev} type SMI parent mlx5_${dev}

#    guid=$(ibstat smi${dev} 2>/dev/null | awk '/Port GUID:/ {print $3; exit}')

#    if [ -n "$guid" ]; then
#        echo "Starting opensm for smi${dev} (GUID: $guid)"
#        opensm -g "$guid" >/tmp/opensm_${dev}.log 2>&1 &
#    else
#        echo "Failed to get GUID for smi${dev}"
#    fi
#done
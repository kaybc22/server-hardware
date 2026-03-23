#!/bin/bash

# List of mlx5 parent device numbers
devices=(0 1 6 11 4 5 14 15)

# Base command path
CMD="/opt/mellanox/iproute2/sbin/rdma"

for dev in "${devices[@]}"; do
    echo "Adding smi-mlx5_${dev}..."
    sudo $CMD dev add smi-mlx5_${dev} type SMI parent mlx5_${dev}
done

echo "All RDMA SMI devices added."


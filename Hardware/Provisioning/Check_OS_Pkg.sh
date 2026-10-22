#!/bin/bash


echo "Checking OS Utilities..."
apt list | egrep -i "^(fio|sysstat|nvme-cli|sshpass|ipmitool|dos2unix|infiniband-diags|make|gcc|hwloc|numactl|net-tools|jq|policycoreutils-python-utils|python3-pip|python3-matplotlib)/" 

echo "Checking Conatainer..."
apt list | egrep -i "^(docker.io|docker-compose-v2|docker-ce|docker-ce-cli|containerd.io|docker-buildx-plugin|docker-compose-plugin)/"

echo "Checking DOCA, NVIDIA, CUDA, NCCL, MFT, IB tools..."
apt list | egrep -i "doca-all/|mft/|infiniband-diags|mstflint/|libibumad3|nvidia-open/|nvlink5/|cuda-toolkit-13/|nvidia-fabricmanager/|nvlsm|datacenter-gpu-manager-4-cuda13/|libnccl-dev|libnccl2|nvidia-container-toolkit"

#sudo apt purge $(dpkg -l | awk '/^rc/ {print $2}') -y

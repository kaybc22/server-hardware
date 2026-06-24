#!/bin/bash

echo "Removing DOCA, NVIDIA, CUDA, NCCL, MFT, IB tools..."
sudo apt remove -y \
doca-all mft infiniband-diags mstflint libibumad3 \
nvidia-open  cuda-toolkit-13-1 nvidia-fabricmanager nvlsm \
datacenter-gpu-manager-4-cuda13 libnccl-dev libnccl2 \
nvidia-container-toolkit

for f in $( dpkg --list | grep -E 'doca|flexio|dpa-gdbserver|dpa-stats|dpa-resource-mgmt|dpaeumgmt' | awk '{print $2}' ); do echo $f ; sudo apt remove --purge $f -y ; done
for f in $(dpkg --list | egrep -i "doca|cuda|datacenter|nvidia|nvlsm|nvlink5" | awk '{print $2}'); do echo $f ; apt remove --purge $f -y ; done; apt autoremove -y; apt clean

echo "Removing Docker and container tools..."
sudo apt remove -y \
docker.io docker-compose-v2 docker-ce docker-ce-cli containerd.io \
docker-buildx-plugin docker-compose-plugin

echo "Removing utilities and monitoring tools..."
sudo apt remove -y \
fio sysstat nvme-cli sshpass ipmitool dos2unix make gcc hwloc numactl \
net-tools jq policycoreutils-python-utils python3-pip \
python3-matplotlib

echo "Cleaning up unused dependencies..."
sudo apt autoremove -y
sudo apt purge $(dpkg -l | awk '/^rc/ {print $2}') -y

echo "Done."


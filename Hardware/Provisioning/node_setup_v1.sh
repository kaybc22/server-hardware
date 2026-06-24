#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[31m=== the system Will reboot for the new kernel update ===\033[0m"
sleep 5
echo -e "\033[33m=== NVIDIA + DOCA + Docker + Utilities Setup ===\033[0m"

### -------------------------------
### 1. Add Required Repositories
### -------------------------------

echo -e "\033[33m[1/6] Adding CUDA repo...\033[0m"
distribution=$(. /etc/os-release; echo ${ID}${VERSION_ID} | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

echo -e "\033[33m[2/6] Adding DOCA repo...\033[0m"
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.2.0/ubuntu24.04/x86_64/"
curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub \
    | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" \
    > /etc/apt/sources.list.d/doca.list

echo -e "\033[33m[3/6] Adding Docker repo...\033[0m"
apt install -y ca-certificates
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" \
    > /etc/apt/sources.list.d/docker.list

echo -e "\033[33m[4/6] Adding NVIDIA container toolkit repo...\033[0m"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo -e "\033[33m[5/6] Updating apt...\033[0m"
apt update -y

### -------------------------------
### 2. Install Packages
### -------------------------------

echo -e "\033[33m[6/6] Installing utilities...\033[0m"
apt install -y \
    fio sysstat nvme-cli sshpass ipmitool dos2unix infiniband-diags libibumad3 \
    make gcc hwloc numactl net-tools mstflint pv powertop nload iftop unzip \
    expect dkms jq nfs-common python3 openmpi-bin cifs-utils nmon \
    policycoreutils-python-utils python3-pip python3-matplotlib

echo -e "\033[33mInstalling Docker...\033[0m"
apt install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

echo -e "\033[33mInstalling NVIDIA + DOCA...\033[0m"
apt install -y doca-all cuda-toolkit-13 nvlsm \
    nvidia-fabricmanager nvidia-container-toolkit

echo -e "\033[33mInstalling NCCL + DCGM...\033[0m"
apt install -y datacenter-gpu-manager-4-cuda13 libnccl-dev libnccl2

echo -e "\033[33mInstalling GPU driver....\033[0m"
sleep 5
apt install -y nvidia-open

### -------------------------------
### 3. Environment Setup
### -------------------------------

echo -e "\033[33mConfiguring environment variables...\033[0m"
modprobe ib_umad
echo 'export PATH=/usr/local/cuda-13.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
echo "ib_umad" > /etc/modules

### -------------------------------
### 4. Enable Services
### -------------------------------

echo -e "\033[33mEnabling NVIDIA services...\033[0m"
systemctl enable nvidia-fabricmanager
#systemctl start nvidia-fabricmanager

systemctl enable nvidia-dcgm
#systemctl start nvidia-dcgm

echo -e "\033[33mConfiguring NVIDIA container runtime...\033[0m"
nvidia-ctk runtime configure --runtime=docker
#systemctl restart docker

echo -e "\033[33m=== Installation Complete ===\033[0m"
#echo -e "\033[33mReboot the system for the new kernel update\033[0m"
echo -e "\033[33mReload your shell or run: source ~/.bashrc\033[0m"

sleep 5
reboot


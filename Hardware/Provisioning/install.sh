#!/bin/bash

# Optimized Automation Script for Bare Metal Server Setup
# Installs NVIDIA GPU, DOCA, MFT, Fabric Manager, DCGM, NCCL, Docker, and NVIDIA Container Toolkit.
# Combines repo additions and minimizes apt update calls.
# Run as root or with sudo. Assumes Ubuntu 22.04 (based on DOCA repo).
# Usage: sudo bash setup-automation.sh

set -e  # Exit on any error

# Install base packages
echo "Installing base packages..."
apt update -y
apt install -y fio sysstat nvme-cli sshpass ipmitool dos2unix infiniband-diags libibumad3 make gcc hwloc numactl net-tools mstflint pv powertop nload iftop unzip dos2unix expect dkms jq nfs-common python3 openmpi-bin cifs-utils nmon policycoreutils-python-utils python3-pip python3-matplotlib
sleep 1

# Combine repository additions (NVIDIA CUDA, DOCA, Docker, NVIDIA Container Toolkit)
echo "Setting up repositories..."
# NVIDIA CUDA repo
distribution=$(. /etc/os-release; echo $ID$VERSION_ID | sed -e 's/\.//g')
wget -q https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb  # Cleanup

# DOCA repo
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu22.04/x86_64/"
curl -s https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list

# Docker repo
apt install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# NVIDIA Container Toolkit repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Single apt update after all repos are added
echo "Updating package lists..."
apt update -y

# Install NVIDIA and CUDA packages
echo "Installing NVIDIA and CUDA packages..."
apt install -y nvidia-open cuda-toolkit-13-0

# Enable CUDA environment
echo "Enabling CUDA environment..."
echo "export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
echo "ib_umad" > /etc/modules
source ~/.bashrc

# Install DOCA
echo "Installing DOCA..."
apt install -y doca-all

# Install MFT (NVIDIA FW management tools)
echo "Installing MFT..."
wget -q https://www.mellanox.com/downloads/MFT/mft-4.33.0-169-x86_64-deb.tgz
tar -xvf mft-4.33.0-169-x86_64-deb.tgz
cd mft-4.33.0-169-x86_64-deb
./install.sh
mst start
mst status -v
cd ..
rm -rf mft-4.33.0-169-x86_64-deb.tgz mft-4.33.0-169-x86_64-deb  # Cleanup

# Install additional NVIDIA tools
apt install -y nvlsm nvlink5 mstflint

# Enable NVIDIA Fabric Manager
echo "Enabling NVIDIA Fabric Manager..."
systemctl enable nvidia-fabricmanager
systemctl start nvidia-fabricmanager

# Install and start DCGM
echo "Installing and starting DCGM..."
apt install -y ddatacenter-gpu-manager-4-cuda13
systemctl --now enable nvidia-dcgm
systemctl start nvidia-dcgm
dcgmi discovery -l

# Install and build NCCL tests
echo "Installing and building NCCL tests..."
apt install -y libnccl-dev libnccl2
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make src.build  # Builds without MPI; for MPI, run 'make MPI=1' separately if needed
cd ..

# Install Docker
echo "Installing Docker..."
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Verify Docker installation
docker run hello-world

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Setup complete! Reboot recommended for full effect."
echo "To run NCCL tests with MPI: cd nccl-tests && make MPI=1"

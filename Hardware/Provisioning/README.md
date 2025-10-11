OS installation --> manage the OS Utilities --> drivers, libraries and dependencies --> tools --> container, working environment


#Check_system_fw.sh --> To check the system and GPU components Firmware via Redfish
#GPU_FW_update.sh --> Update GPU FW 
#Redfish_managment.txt --> URI
#BF3_FW_update.sh --> Update BF3 FW

#OS installation
+boot
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux 5.15.0-136-generic'
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux 5.15.0-144-generic'
sudo reboot

#manage the OS Utilities
apt-high level:/etc/apt/sources.list vs /var/lib/dpkg/status #dpkg low level
sudo dpkg --configure -a # reconfig the interupted package
sudo apt --fix-broken install# re-install with checkign the dependencies
find /var/cache/apt/archives/ -name "ucx*.deb"

dpkg -L infiniband-diags
dpkg -S $(which powertop)

#remove unused/copnflicted packages
sudo dpkg --remove doca-cx-libs
sudo dpkg -i --force-overwrite /path/to/doca-cx-libs_2.6.0058-1_amd64.deb
sudo dpkg --configure -a
sudo apt-get update
sudo apt-get -f install
/etc/apt/sources.list
/etc/apt/sources.list.d/doca.list

sudo dpkg --configure -a
dpkg -i  abc.deb
dpkg -r abc
sudo apt-get autoremove

dpkg-deb -c abs.deb
dpkg -l


apt-cache show mft
sudo apt-get install ibutils -y
sudo less /var/log/apt/history.log

type mget_temp
dpkg -S /usr/bin/mget_temp
mft: /usr/bin/mget_temp
mget_temp -d mlx5_14

ldd --version

###waiting for the cach lock: could not get lock /var/lib/dpkg/lock-frontend
sudo lsof /var/lib/dpkg/lock-frontend
sudo rm /var/lib/dpkg/lock-frontend

ps aux | grep apt
vim /etc/apt/sources.list
ls /etc/apt/sources.list.d/
sudo dpkg -r cuda-keyring
sudo dpkg --purge cuda-keyring

apt-cache depends doca-all
apt-cache show doca-all


nm /lib/modules/$(uname -r)/updates/dkms/nvidia-modeset.ko | grep nvidia_get_rm_ops
objdump -d /lib/modules/$(uname -r)/updates/dkms/nvidia-modeset.ko | less


#drivers, libraries and dependencies
install all the Tools and SW package for GPU (driver, cuda, nvlsm, nvlink,...) --> running some tests (dcgm, nccl, fd, nvqual,...) to confirm the GPU and system function properly --> install monitoring tools
(dcgm-exporter, node-exporter, prometheus, grafana) --> running some application (llm, pytorch, mlperf...) to capture the peformance --> reviewing the numa nodes, pcie link status, cpu and mem usage to study
the system behavior. --> Cluster testing (slurm, RDMA, nccl, Openmpi...)  
 
#uninstall all the previous installation packages
for f in $(dpkg --list | egrep "doca|cuda|datacenter|nvidia|nvlsm|nvlink5" | awk '{print $2}'); do echo $f ; apt remove --purge $f -y ; done
 apt autoremove -y; apt clean


##mlcommons
apt install -y  fio sysstat nvme-cli sshpass ipmitool dos2unix infiniband-diags libibumad3 make gcc hwloc numactl net-tools mstflint pv powertop nload iftop unzip dos2unix expect dkms jq nfs-common python3 openmpi-bin cifs-utils nmon policycoreutils-python-utils python3-pip python3-matplotlib 

#driver
wget https://us.download.nvidia.com/tesla/580.65.06/NVIDIA-Linux-x86_64-580.65.06.run
 
##B200 prerequitesite packages
apt install -y  fio sysstat nvme-cli sshpass ipmitool dos2unix infiniband-diags libibumad3 make gcc hwloc numactl net-tools mstflint pv powertop nload iftop unzip dos2unix expect jq linux-tools-common

#add the NV working repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g'); wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb; sudo dpkg -i cuda-keyring_1.1-1_all.deb; apt update

##Alternative way to add the public key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub; sudo mv 3bf863cc.pub /etc/apt/trusted.gpg.d/nvidia-cuda.asc

#check the requirement packages
apt list | egrep -i "nvidia-open|cuda-toolkit-13-0|nvlsm|nvlink|fabricmanager"
apt list | egrep -i "doca-all|mft|infiniband-diags|mstflint|libibumad3"
 
#install the driver and tools
apt install -y nvidia-open (or cuda-drivers) # nvshmem
apt install -y  cuda-toolkit-13-0 
apt install -y nvlsm nvlink5 
+sudo apt-mark unhold nvidia-persistenced; sudo apt-mark unhold nvidia-persistenced; apt-mark showhold; apt install -y nvidia-persistenced=570.148.08-1ubuntu1


#ucx issue
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvlink5_575.57.08-1_amd64.deb
dpkg -i nvlink5_575.57.08-1_amd64.deb
apt --fix-broken install
sudo dpkg -i --force-overwrite /var/cache/apt/archives/libnvsdm_580.82.07-1_amd64.deb

 
#enable the working environment
echo "export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}" >> ~/.bashrc; echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc; echo "ib_umad" > /etc/modules; source ~/.bashrc
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nouveau.conf"; sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nouveau.conf"; sudo update-initramfs -u
echo "ib_umad" > /etc/modules (openibd)

echo "export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}" >> ~/.bashrc; echo "export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc; echo "ib_umad" > /etc/modules; source ~/.bashrc


#mft NV FW management tools (mft)
wget https://www.mellanox.com/downloads/MFT/mft-4.33.0-169-x86_64-deb.tgz; tar -xvf mft-4.33.0-169-x86_64-deb.tgz; cd mft-4.33.0-169-x86_64-deb; ./install.sh; mst start; mst status -v
mft-4.30.1-1210-x86_64-deb.tgz
mft-4.31.0-149-x86_64-deb.tgz
mft-4.32.0-120-x86_64-deb.tgz
mft-4.33.0-169-x86_64-deb.tgz
wget https://www.mellanox.com/downloads/MFT/mft-4.30.1-1210-x86_64-deb.tgz; tar -xvf mft-4.30.1-1210-x86_64-deb.tgz; cd mft-4.30.1-1210-x86_64-deb; ./install.sh; mst start; mst status -v


#doca - install DOCA
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/2.9.2/ubuntu22.04/x86_64/"; curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub; echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list; sudo apt-get update
apt install -y doca-all
https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu22.04/x86_64/
https://linux.mellanox.com/public/repo/doca/3.0.0/ubuntu24.04/x86_64/
https://linux.mellanox.com/public/repo/doca/2.10.0/ubuntu22.04/x86_64/
https://linux.mellanox.com/public/repo/doca/2.9.2/ubuntu22.04/x86_64/
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu22.04/x86_64/"

nvidia-smi --query-gpu=index,name,serial,uuid,driver_version,vbios_version,pstate,pcie.link.width.max --format=csv,noheader

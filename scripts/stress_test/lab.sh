#!/bin/bash

# Duration in seconds (1 hour)
DURATION=3600

#install Linux utilities
install_utls() {
  apt update
  apt install -y  fio sysstat nvme-cli sshpass ipmitool dos2unix infiniband-diags libibumad3 make gcc hwloc numactl net-tools mstflint pv powertop nload iftop unzip dos2unix expect jq linux-tools-common
#  distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g'); wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb; sudo dpkg -i cuda-keyring_1.1-1_all.deb; apt update

}

# Stress test function
run_stress_tests() {
  echo "===== Running stressapptest ====="
  stressapptest -W -s $DURATION -M $(($(free -m | awk '/Mem:/ {print int($2 * 0.8)}'))) -m 8

  echo "===== Running fio on NVMe ====="
#  fio --name=nvme_test --filename=/dev/nvme0n1 --rw=randwrite --bs=4k --size=1G --numjobs=4 --time_based --runtime=$DURATION --group_reporting

  echo "===== Running Python CPU stress ====="
  python3 - <<EOF
import time
import threading
def burn():
   while True:
       x = 2 ** 1024
threads = []
for _ in range(8):
   t = threading.Thread(target=burn)
   t.daemon = True
   t.start()
time.sleep($DURATION)
EOF

  echo "===== Running stress-ng ====="
  stress-ng --cpu 8 --io 4 --vm 2 --vm-bytes 1G --timeout ${DURATION}s --metrics-brief

  echo "===== Stress Test Complete ====="
}

# nccl GPU test
nccl() {
  installed=$(apt list --installed 2>/dev/null | grep -i nccl | grep '2.28.3-1+cuda13.0')
  version=$(apt list --installed 2>/dev/null | grep -i '^libnccl2/' | awk -F'/' '{print $2}' | awk '{print $1}')
# strings /usr/lib/x86_64-linux-gnu/libnccl.so | grep  "NCCL version"
  if [ -n "$version" ]; then
    echo "nccl version $version"
  else
    echo 'Please install the preferred version'
    #sudo apt install -y libnccl-dev libnccl2 -y; git clone https://github.com/NVIDIA/nccl-tests.git; cd nccl-tests; make
  fi
  #./build/all_reduce_perf -b 8 -e 32G -f 2 -t 8
  #./build/alltoall_perf -b 8 -e 32G -f 2 -t 8
}

# nvbandwidth
nvbandwidth() {
  apt install libboost-program-options-dev -y; git clone https://github.com/NVIDIA/nvbandwidth; cd $(pwd)/nvbandwidth; sudo ./debian_install.sh; cmake .; make
  ./nvbandwidth
}

#DCGM
dcgm () {
   sudo apt-get install -y datacenter-gpu-manager-4-cuda13
   sudo systemctl --now enable nvidia-dcgm; sudo systemctl start nvidia-dcgm; dcgmi discovery -l
#enable the GPU persistent mode
   #for i in {0..7}; do nvidia-smi -i $i -pl 1000; done
   #for i in {0..7}; do nvidia-smi -i $i -pm 1; done
   #time dcgmi diag -r 4 & nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,temperature.gpu.tlimit,power.draw,power.draw.average,utilization.gpu,utilization.memory,clocks_throttle_reasons.active --format=csv -l 1 -f $(pwd)/gpu_usage_$(date +%s%3N).log &
   #time dcgmi diag --run diagnostic,memtest,targeted_power --parameters targeted_power.test_duration=500 --iterations 2 & nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,temperature.gpu.tlimit,power.draw,utilization.gpu,utilization.memory,clocks_throttle_reasons.active --format=csv -l 1 -f $(pwd)/gpu_usage$(date +%Y%m%d_%H%M%S).log &
}

hardware_info() {
# Display system hardware info
echo -e "\033[32m===== SYSTEM HARDWARE INFO =====\033[0m"
echo ""
echo -e "\033[32m=====System Product vendor=====\033[0m"
dmidecode -t baseboard | egrep  "Manufacturer|Product"
dmidecode -t bios | grep -v "#" |grep -iE "present|vendor|version|release|size"
echo ""
echo -e "\033[32m=====System CPU Info=====\033[0m"
lscpu | egrep -i "core|numa|mib|mhz|model"
#numactl --hardware
echo ""
echo -e "\033[32m=====System Mem Info=====\033[0m"
echo "There are toltal $(dmidecode -t memory | grep -i "form factor" | wc -l) memory sticks"
free -h
dmidecode -t memory | egrep -i "gb|mt|ddr|Manufacturer" | head -5
echo ""
echo -e "\033[32m=====System Block devices=====\033[0m"
lsblk
nvme list
ls -l --color /dev/disk/by-path/ | grep -v '\-part' | sort -k11 | awk '{ print $9 $10 $11}'
echo ""
echo -e "\033[32m=====System OS info=====\033[0m"
cat /etc/*release | egrep -i "DISTRIB|VERSION"
echo ""
echo -e "\033[32m=====System uptime info=====\033[0m"
uptime
echo ""
echo -e "\033[32m=====GPU Network components=====\033[0m"
lspci | egrep -i "nvidia|mella"
echo ""
echo -e "\033[32m================================\033[0m"
echo ""
echo -e "\033[32m=====PCIe devices and Bus_ID=====\033[0m"
lspci | egrep -i "mella|nvidia|nvme|volatile|eth"
echo ""
echo -e "\033[32m================================\033[0m"

}

#Update nad install the Utls
#install_utls

#call hardware info function
hardware_info

# Call the stress test function
#run_stress_tests

#call nccl function
#nccl

#call nvbandwidth function
#nvbandwidth

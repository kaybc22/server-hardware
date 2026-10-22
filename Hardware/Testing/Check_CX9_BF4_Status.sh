#!/bin/bash

mst start >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    echo "MST started successfully"
else
    echo "MST failed with exit code $rc"
fi
#mst status -v | awk '/ConnectX|BlueField/ && $3 ~ /\.0$/ {print $4}' #only grep 56:00.0 not 56:00.1
mst status -v | grep -iv "Pure_PCIe" | awk '/ConnectX|BlueField/ && $3 ~ /\.0$/ {print $0}'
#enable ethernet ports
for i in $(ibdev2netdev | awk '{ print $5}'); do ifconfig $i up; done

#check Port and link status of CX9 and BF4
echo -e "\033[32mChecking Ports and links status\033[0m"
for i in $(mst status -v | grep -iv "Pure_PCIe" |awk '/ConnectX|BlueField/ && $3 ~ /\.0$/ {print $4}'); do ibstat $i | grep -iE "mlx5|state:|Link layer:|Port GUID:|lid:"; done

#Check cables
echo -e "\033[32mChecking cables status\033[0m"
for i in $(mst status -v | grep -iv "Pure_PCIe"| awk '/ConnectX|BlueField/ && $3 ~ /\.0$/ {print $4}'); do echo $i; mlxlink -d $i -m  | grep -iE "Cable|Opcode|Speed"; done
#
echo -e "\033[32mChecking RDMA Device\033[0m"
rdma dev show
echo -e "\033[32mChecking RDMA Link\033[0m"
rdma link show #show_gids -->  ip -br addr show enp86s0f0np0

echo -e "\033[32mChecking Numa mapping\033[0m"
#mst status -v | grep -iv "Pure_PCIe" | awk '/ConnectX|BlueField/ && $3 ~ /\.0$/ {print $0}'
numa_mapping () {
for d in /sys/class/infiniband/*; do
    [ -e "$d" ] || continue

    dev=$(basename "$d")
    numa=$(cat "$d/device/numa_node" 2>/dev/null)

    [ -n "$numa" ] && [ "$numa" -ge 0 ] 2>/dev/null || {
        echo "$dev NUMA node: unknown"
        continue
    }

    cpus=$(lscpu -p=CPU,NODE | awk -F, -v n="$numa" \
        '$1 !~ /^#/ && $2 == n {print $1}')

    if [ -n "$cpus" ]; then
        min=$(echo "$cpus" | sort -n | head -1)
        max=$(echo "$cpus" | sort -n | tail -1)
        echo "$dev NUMA node: $numa CPUs: $min to $max"
    else
        echo "$dev NUMA node: $numa CPUs: none"
    fi
done
}
numa_mapping

#taskset -c 2 numactl ib_write_bw --ib-dev=mlx5_0 --ib-port=1 --port=18516 --CPU-freq --bidirectional --qp=4 --duration=3 --report_gbits & sleep 1; taskset -c 126 ib_write_bw --ib-dev=mlx5_2 --ib-port=1 --port=18516 --CPU-freq --qp=4 --bidirectional --duration=3 --report_gbits 127.0.0.1
#taskset -c 4 ib_write_bw -d mlx5_0 --ib-port=1 & taskset -c 88 ib_write_bw -d mlx5_2 --ib-port=1 127.0.0.1 #test accross CPU
#taskset -c 4 ib_write_bw -d mlx5_0 --ib-port=1 & ib_write_bw -d mlx5_0 --ib-port=1 127.0.0.1 #test port

#MODULE_VOLTAGE_MT4133_PCICONF2  
#sudo mlxlink -d /dev/mst/mt4133_pciconf2 --cable --ddm | grep '^Voltage' | head -1 | awk '{print $3}'
#GRACE_TEMPERATURE_CPU3 - Player 8 - INFO 
#cat /sys/class/thermal/thermal_zone3/temp
#HOST_NETWORK_PHY_BER_MT4133_PCICONF0 - Player 0 - INFO 
#sudo mlxlink -d /dev/mst/mt4133_pciconf0 -c | grep 'Effective Physical BER\|Raw Physical BER' | grep -v 'Per Lane'
#Effective Physical BER             : 15E-255
#Raw Physical BER                   : 7E-9
#HOST_MLXLINK_COUNTER - Player 3 - INFO 
#sudo mlxlink -d 0006:01:00.0 --port_type PCIe --pcie_index 0 --node 0 --depth 0 -c
#sudo mget_temp -d /dev/mst/mt4133_pciconf6
# BF_SSD_TEMPERATURE_NVME0N1 - Player 8 - INFO 
#sudo nvme smart-log /dev/nvme0n1 | grep temperature | awk '{print $3}' #'critical_warning' $2

#[14/08/2026 19:38:33] PORTS_CONFIGURATION - INFO 
#[run on local host]: sudo /opt/mellanox/iproute2/sbin/ip route add local 192.168.2.2 dev eth15 src 192.168.2.2 table 13
#sudo /opt/mellanox/iproute2/sbin/ip addr flush dev eth15
#sudo /opt/mellanox/iproute2/sbin/ip addr add 192.168.2.2/24 dev eth15
#sudo /opt/mellanox/iproute2/sbin/ip link set eth15 up
#sudo /opt/mellanox/iproute2/sbin/ip rule add iif eth15 lookup local proto kernel pref 0
#sudo /opt/mellanox/iproute2/sbin/ip route add local 192.168.2.2 dev eth15 src 192.168.2.2 table 13
#sudo /opt/mellanox/iproute2/sbin/ip -br link show dev eth15 | awk '{print $2}'

#keywords
#[psa_stressor] Stress started # [host_mlxlink_counter]
#19:38:36 Starting psa_stressor on # Starting host_mlxlink_counter
#Task _get_player_run_list_4 has completed successfully
#is_device_in_version_range

#[14/08/2026 19:38:16] PLAYER_8 - INFO
#[14/08/2026 19:38:33] PORTS_CONFIGURATION - INFO 
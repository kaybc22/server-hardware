#!/bin/bash

function Get_NIC_Firmware(){
for i in {0..10}; do flint -d  /dev/mst/mt41692_pciconf$i q | egrep -i "psid|fw"; done
#echo "y" | mlxfwmanager --online -u
#lspci | grep -i mella
}

function Get_NIC_Info(){
for i in $(mst status -v | awk '/ConnectX|BlueField3/ {print$3}' | grep -v '\.1$'); do echo $i; sudo lspci -vvs $i | egrep "V2|V3|SN"; done
}

function Get_NIC_Info_enhance() {
     local IFS=$' \t\n'
     for i in $(mst status -v | awk '/BlueField3|ConnectX7/ {print$2 "   " $3}'|  grep -v '\.1$') ; do 
	 echo $i 
	 #local bus_id="${i%% *}" 
	 sudo lspci -vvvs $i | egrep "V2|V3|SN"; done
     #for i in $(mst status -v | awk '/BlueField3|ConnectX7/ {print$2 "   " $3}'|  grep -v '\.1$') ; do echo $i ; bus_id=$(echo $i | grep -v 'mst') ; sudo lspci -vvvs $bus_id | egrep "V2|V3|SN"; done
	 #mst status -v | awk '/BlueField3|ConnectX7/ {print $2 " " $3}' | grep -v '\.1$' | awk '{print $2}'
     #bus_id=$(echo $i | grep -v mst')
	 #for i in $(mst status -v | awk '/BlueField3|ConnectX7/ {print $2 " " $3}' | grep -v '\.1$'); do echo $i; bus_id=$(echo $i | cut -d' ' -f2); sudo lspci -vvvs $bus_id | egrep "V2|V3|SN"; done
     #sudo mlxlink -d $i | egrep "State|Recommendation"
}


function Check_NIC_Cable() {
#No Bus ID
for i in $(mst status -v | awk '/BlueField3|ConnectX7|ConnectX6/  {print$2 ":" $3}'); do  echo $i; sudo mlxlink -d "${i%%:*}" | egrep "State|Recommendation";  done
}
for i in $(mst status -v | awk '/BlueField3|ConnectX7|ConnectX6/  {print$2 ":" $3}'); do  echo ${i%%:*};  done

function Check_NIC_Cable_Bus_ID() {
#List of devices -- not working yet
   devices=("BlueField3" "ConnectX7")
   for device_id in "${devices[@]}"; do
          devices_info=$(mst status -v | awk '/'"$device_id"'/ {print $2, $3}')
          List_devices=$(mst status -v | awk '/'"$device_id"'/ {print $2}') 
          devices_state=$(sudo mlxlink -d "$List_devices" | egrep -o "State|Recommendation")
          echo  "Device Info: $devices_info"
          echo  $devices_state
  done
}

function Get_rshim_Bus(){
for i in $(ls /dev/ | grep -i rsh) ; do echo $i ; sudo cat /dev/$i/misc | grep DEV; done
}

function Get_GPU_Info() {
cat /proc/driver/nvidia/version
nvidia-smi -q | egrep "Name|000000|VBIOS Version|Serial Number|Bus|Board Part"
}

while true; do
    echo "Select an option:"
    echo "1. Get NIC Info"
    echo -e "\033[33m2. Check the NIC cable\033[0m"
    echo "3. Get GPU Info"
    echo "4. Get NIC device and bus ID.. not complete yet"
    echo -e "\033[33m5. Back to the main....\033[0m"

    # Read user input
    read -p "Enter your choice: " choice

    # Execute the selected function
    case $choice in
        1) Get_NIC_Info ;;
        2) Check_NIC_Cable ;;
        3) Get_GPU_Info ;;
        4) Get_NIC_Info_enhance && Get_rshim_Bus ;;
        5) echo "Exiting..." ; break ;;
        *) echo "Invalid choice. Please select a valid option." ;;
    esac
done



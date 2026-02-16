#!/usr/bin/bash

function Check_BF3_CX7_mode() {
for i in $(mst status -v | awk '/ConnectX|BlueField3/ {print$2}' | grep -v '\.1$'); do mlxconfig -d $i -e q INTERNAL_CPU_OFFLOAD_ENGINE INTERNAL_CPU_MODEL; done
}

function DPU_mode() {
for i in $(mst status -v | awk '/BlueField3/ {print$2}' | grep -v '\.1$');do echo "y" | mlxconfig -d $i s INTERNAL_CPU_OFFLOAD_ENGINE=0; done
}

function NIC_mode() {
for i in $(mst status -v | awk '/BlueField3/ {print$2}' | grep -v '\.1$'); do echo "y" | mlxconfig -d $i s INTERNAL_CPU_OFFLOAD_ENGINE=1; done
}

function Check_IB_ETH_mode() {
for i in $(mst status -v | awk '/ConnectX|BlueField3/ {print$2}' | grep -v '\.1$'); do mlxconfig -d $i -e q LINK_TYPE_P1; done
}

function IB_mode() {
for i in $(mst status -v | awk '/ConnectX|BlueField3/ {print$2}' | grep -v '\.1$'); do echo "y" | mlxconfig -d $i s LINK_TYPE_P1=1; done
}

function ETH_mode() {
for i in $(mst status -v | awk '/ConnectX|BlueField3/ {print$2}' | grep -v '\.1$'); do echo "y" | mlxconfig -d $i s LINK_TYPE_P1=2; done
}


while true; do
    echo "Select an option:"
    echo -e "\033[33m1. Check BF3 Mode...\033[0m"
    echo "2. Change to DPU mode"
    echo "3. Change to NIC mode"
    echo -e "\033[33m4. Check ETH or IB mode for BF3\033[0m"
    echo -e "\033[33m5. Change Link Layer to IB mode\033[0m"
    echo "6. Change Link Layer to ETH mode"
    echo "7. Exit"

    # Read user input
    read -p "Enter your choice: " choice

    # Execute the selected function
    case $choice in
        1) Check_BF3_CX7_mode ;;
        2) DPU_mode ;;
        3) NIC_mode ;;
        4) Check_IB_ETH_mode ;;
        5) IB_mode ;;
        6) ETH_mode ;;
        7) echo "Exiting..." ;  break ;;
        *) echo "Invalid choice. Please select a valid option." ;;
    esac
done		
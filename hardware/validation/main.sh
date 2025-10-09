#!/bin/bash

# Define functions
function Get_GPU_NIC_Info() {
    /opt/testing/utls/Get_GPU_NIC_Info.sh
}

function Check_BF3_mode() {
    /opt/testing/utls/Check_BF3_mode.sh
}

function install_doca_utls() {
    /opt/testing/utls/install_doca_utls.sh
}

function run_nvqual() {
    /opt/testing/utls/thermal.sh
}
function Run_Cpu_Nvme_Temp() {
    /opt/testing/utls/Run_Cpu_Nvme_Temp.sh
}
function function10() {
    echo "Running Function "
}
function function11() {
    echo "Running Function "

}
function function12() {
    echo "Running Function "
}


# Display menu
while true; do
    #echo -e "\033[31mNoted: Please run -mst start- to Loading MST PCI module...\033[0m"
    echo "Select an option:"
    echo "1. Get GPU & NIC Info"
    echo -e "\033[32m2. Check BF3 Mode\033[0m"
    echo "3. Install Doca and Utls"
    echo -e "\033[33m4. Run Nvqual...\033[0m"
    echo "5. Stress Cpu and NVMe..."
    # echo "7. Run Function 1"
    # echo "8. Run Function 2"
    # echo "9. Run Function 3"
    # echo "10. Run Function 1"
    # echo "11. Run Function 2"
    # echo "12. Run Function 3"
    echo "6. Exit"

    # Read user input
    read -p "Enter your choice: " choice

    # Execute the selected function
    case $choice in
        1) Get_GPU_NIC_Info ;;
        2) Check_BF3_mode ;;
        3) install_doca_utls ;;
        4) run_nvqual ;;
        5) Run_Cpu_Nvme_Temp ;;
        # 7) Reserved ;;
        # 8) Reserved ;;
        # 9) Reserved ;;
        # 10) Reserved ;;
        6) echo "Exiting..." ; break ;;
        *) echo "Invalid choice. Please select a valid option." ;;
    esac
done
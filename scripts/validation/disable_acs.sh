#!/bin/bash
for pcie_dev in $(lspci -d "*:*:" |awk '{print$1}');do
        setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w >/dev/null 2>&1
        if [ $? -ne 0 ];then
		        echo "setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w" | tee -a check_function.txt
                echo "${pcie_dev} does not support ACS, skipping"
                continue
        fi
        #generate_log_file "Disabling ACS on $(lspci -s ${pcie_dev})"
        setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w=0000 
        if [ $? -ne 0 ];then
		        echo "setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w=0000" | tee -a dis_fuction.txt
                echo "ERROR: Failed to disable ACS on $(lspci -s ${pcie_dev})"
                exit_code=1
                continue
        fi
        new_val=$(setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w |awk '{print$NF}')
        if [ "${new_val}" != "0000" ];then
                echo "ERROR: Failed to disable ACS on $(lspci -s ${pcie_dev})"
				echo "setpci -v -s ${pcie_dev} ECAP_ACS+0x6.w |awk '{print$NF}'" | tee -a chk_new.txt
                exit_code=1
        fi
done
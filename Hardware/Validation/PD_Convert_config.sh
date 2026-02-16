#!/bin/bash

X14_convert() {
read -p "Input the reference file with the complet path: /opt/configfile.. " Ref_file
sed -i 's/mlx5_0/mlx5_7/g; s/mlx5_1/mlx5_8/g; s/mlx5_2/mlx5_9/g; s/mlx5_3/mlx5_10/g; s/mt4129_pciconf0.2/mt4129_pciconf7.2/g; s/mt4129_pciconf0.3/mt4129_pciconf7.3/g' $Ref_file
grep --color=auto -E "mlx5|mt4129_pciconf" "$Ref_file"
echo -e "\033[33mX14 mlx5_ onboard device start from mlx5_7,8,9,10\033[0m"
}

X13_convert() {
read -p "Input the reference file with the complet path: /opt/configfile.. " Ref_file
sed -i 's/mlx5_7/mlx5_0/g; s/mlx5_8/mlx5_1/g; s/mlx5_9/mlx5_2/g; s/mlx5_10/mlx5_3/g; s/mt4129_pciconf7.2/mt4129_pciconf0.2/g; s/mt4129_pciconf7.3/mt4129_pciconf0.3/g' $Ref_file
grep --color=auto -E "mlx5|mt4129_pciconf" $Ref_file
echo -e "\033[31mX14 mlx5 onboard device start from mlx5_0,1,2,3\033[0m"
}


while true; do
    echo -e "\033[33mSelect an option: \033[0m"
    echo "1. Convert the reference configuration to X14"
    echo -e "\033[31m2. Convert the reference configuration to X13\033[0m"
    echo "3. Exit"
    
    # Read user input
    read -p "Enter your choice: " choice

    # Execute the selected function
    case $choice in
        1) X14_convert ;;
        2) X13_convert ;;
        3) echo "Exiting..." ; break ;;
        4) echo "Invalid choice. Please select a valid option." ;;
    esac
done

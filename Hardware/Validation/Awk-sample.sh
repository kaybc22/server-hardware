#!/bin/bash


function Check_Power() {
     echo "Begin comparing power"
     cat $1 | grep "mW" | awk '{ if ($9 > 700000) print $0 }' | tee >(wc -l);;
}
function Check_Temp() {
      echo "Begin comparing temperautre"
      cat $1 | grep "GpuTemp" | awk '{ if ($9 > 90) print "GpuTemp" ":" $9 }' | tee >(wc -l) ;;
}	 
function Check_TLimit() {
      echo "Begin comparing temperautre"
      cat $1| grep "TLimit" | awk '{ if ($9 < 6) print "TLimit" ":" $9 }' | tee >(wc -l) ;;
}


while true; do
    #echo -e "\033[31mNoted: Please run -mst start- to Loading MST PCI module...\033[0m"
	echo "Usage: ./script.sh flie.log "
    echo "Select an option:"
    echo "1. Check and  compare power > 700000 "
    echo -e "\033[32m2. Check and  compare temperautre > 90 in C\033[0m"
    echo "3. Check and  compare Tlimit < 6"
    echo "4. Exit"
	
    read -p "Enter your choice: " option

    case $option in 
    1 Check_Power ;;
	2 Check_Temp ;;
	3 Check_TLimit
	4) echo "Exiting..." ; break ;;
    *) echo "Invalid choice. Please select a valid option." ;;
    esac
done

#cat thermal_test_050724_184114_part_0.log | grep "mW" | awk '{print $9}'
#cat thermal_test_050724_184114_part_0.log | grep "TLimit" | awk '{ if ($7 > 25) print $0 }' | tee >(wc -l)
#cat thermal_test_061824_043113_part_0.log | grep "GpuTemp" | awk '{ if ($9 > 700000) print $0 }' | tee >(wc -l)

#cat thermal_test_050724_184114_part_0.log | grep "mW" | awk '{ if ($9 > 740000) print "PowerMW" ":" $9 }' | tee >(wc -l)
#cat thermal_test_061824_043113_part_0.log | grep "GpuTemp" | awk '{ if ($9 > 90) print "GpuTemp" ":" $9 } | tee >(wc -l)
#cat thermal_test_061824_043113_part_0.log | grep "TLimit" | awk '{ if ($9 < 6) print "TLimit" ":" $9 }' | tee >(wc -l)

#"TLimit" 3 - 32
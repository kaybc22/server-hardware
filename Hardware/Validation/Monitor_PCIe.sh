#!/bin/bash

# Duration: 30 minutes (1800 seconds)
DURATION=1800
INTERVAL=3

# Create single log file
LOGFILE="$(pwd)/PCIe_$(date +"%d-%m-%Y_%H_%M_%S").log"

echo "Starting PCIe monitor at $(date)" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE"
echo "----------------------------------------" >> "$LOGFILE"

# Record start time
START_TIME=$(date +%s)

while (( $(date +%s) - START_TIME < DURATION )); do

    echo "Timestamp: $(date)" >> "$LOGFILE"
    echo "----------------------------------------" >> "$LOGFILE"

    devices=$(lspci | grep -i "3D controller" | awk '{print $1}')

    for bus_id in $devices; do
        echo "Device: $bus_id" >> "$LOGFILE"
        sudo lspci -vvvs "$bus_id" | egrep -w "LnkCap|LnkSta" >> "$LOGFILE"
        echo "" >> "$LOGFILE"
    done

    sleep $INTERVAL
done

echo "Finished at $(date)" >> "$LOGFILE"



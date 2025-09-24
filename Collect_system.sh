#!/bin/bash

# Check if the log file name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <log_file>"
  exit 1
fi

# Define the log file
LOG_FILE=$1

# Define the logging duration (in seconds)
end=$((SECONDS+3600))

# Loop to collect GPU information every 10 seconds
while [ $SECONDS -lt $end ]; do
  date >> $LOG_FILE
  echo "-----GPU info-----" >> $LOG_FILE
  nvidia-smi -q | egrep -iv "N/A|Shutdown|Slowdown|Max|id" | egrep -i 'power draw|minor|bus|Current Temp|t.limit' >> $LOG_FILE
  echo "-----BMC info-----" >> $LOG_FILE
  ipmitool sdr | grep -v ns|  egrep -i 'hgx|fan|inlet' >> $LOG_FILE
  #echo "-----Power reading-----" >> $LOG_FILE
  #ipmitool dcmi power reading >> $LOG_FILE
  #for i in $(lspci | grep -i nvidia | awk '{print $1}'); do echo "$i" >> $LOG_FILE ; lspci -vvv -s $i | egrep -i "DevSta|LnkSta|lnkcap|AERCap" >> $LOG_FILE ; done
  sleep 3  # Interval between each log entry (e.g., 3 seconds)
  #timeout 3600 nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,temperature.gpu.tlimit,power.draw,utilization.gpu,utilization.memory,clocks_throttle_reasons.active --format=csv -l 1 -f /opt/a.log
done
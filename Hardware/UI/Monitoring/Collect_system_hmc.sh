#!/bin/bash

# Check if the log file name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <log_file>"
  exit 1
fi

# Define the log file
LOG_FILE=$1

# Define the logging duration (in seconds)
end=$((SECONDS+4200))

# Loop to collect GPU information every 10 seconds
while [ $SECONDS -lt $end ]; do
  date >> $LOG_FILE
  echo "-----GPU HMC info-----" >> $LOG_FILE
  for i in {1..8}; do
  {
    echo "----- HGX_GPU_SXM_${i} Temp -----"
    curl -skL -u "ADMIN:ADMIN" -X GET "https://172.31.34.74/redfish/v1/Chassis/HGX_GPU_SXM_${i}/Sensors/HGX_GPU_SXM_${i}_TEMP_0" \
      | jq | grep '"Reading":' | head -1
    echo "----- HGX_GPU_SXM_${i} TLimit -----"
    curl -skL -u "ADMIN:ADMIN" -X GET "https://172.31.34.74/redfish/v1/Chassis/HGX_GPU_SXM_${i}/Sensors/HGX_GPU_SXM_${i}_TEMP_1" \
      | jq | egrep -i '"Reading"' | head -1
    echo "----- HGX_GPU_SXM_${i} Power -----"
    curl -skL -u "ADMIN:ADMIN" -X GET "https://172.31.34.74/redfish/v1/Chassis/HGX_GPU_SXM_${i}/Sensors/HGX_GPU_SXM_${i}_Power_0" \
      | jq | grep -iv time | egrep -i '"PeakReading"|"Reading"' | head -2
  } >> "$LOG_FILE"
done
  sleep 1  # Interval between each log entry (e.g., 3 seconds)
  #timeout 3600 nvidia-smi --query-gpu=timestamp,index,temperature.gpu,temperature.memory,temperature.gpu.tlimit,power.draw,utilization.gpu,utilization.memory,clocks_throttle_reasons.active --format=csv -l 1 -f /opt/a.log
done

#172.31.34.74
#169.254.3.254
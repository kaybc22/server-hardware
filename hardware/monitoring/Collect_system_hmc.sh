#!/bin/bash

# Check if the log file name is provided
#if [ -z "$1" ]; then
#  echo "Usage: $0 <log_file>"
#  exit 1
#fi
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <log_file> <IP_address>"
  exit 1
fi

# Define the log file
#LOG_FILE=${1:-"bmc_info_$(date +%Y%m%d_%H%M%S).log"}
LOG_FILE=$1
LOG_FILE=nv_gpu_usage$(date +%Y%m%d_%H%M%S).log
BMC_IP=$2

# Define the logging duration (in seconds)
end=$((SECONDS+3600))

# Loop to collect GPU information every 3 seconds
while [ $SECONDS -lt $end ]; do
  date >> $LOG_FILE
  echo "-----GPU HMC info-----" >> $LOG_FILE
  for i in {1..8}; do curl -skL -u "ADMIN:ADMIN" -X GET https://$BMC_IP/redfish/v1/Chassis/HGX_GPU_SXM_${i}/Sensors/HGX_GPU_SXM_${i}_TEMP_1 | jq | egrep -i "Name|Reading" | head -2 >> $LOG_FILE; curl -skL -u "ADMIN:ADMIN" -X GET https://$BMC_IP/redfish/v1/Chassis/HGX_GPU_SXM_${i}/Sensors/HGX_GPU_SXM_${i}_Power_0 | jq |grep -iv time | egrep -i "Name|PeakReading|reading" | head -3 >> $LOG_FILE; done
  sleep 3
done

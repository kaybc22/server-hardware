#!/bin/bash

LOG_FILE="bmc_info_$(date +%Y%m%d_%H%M%S).log"
INTERVAL=1

echo "Starting BMC logging to $LOG_FILE. Press Ctrl+C to stop."

# Store the entire awk script in a variable to avoid quoting issues.
# This makes it robust and easy to read.
AWK_SCRIPT='
{
  # Remove leading/trailing whitespace
  gsub(/^[ \t]+|[ \t]+$/, "", $0)
  
  # Split the line by the pipe "|" delimiter
  split($0, a, "|")

  # Clean up each part
  name = a[1]
  value = a[2]
  status = a[3]

  gsub(/^[ \t]+|[ \t]+$/, "", name)
  gsub(/^[ \t]+|[ \t]+$/, "", value)
  gsub(/^[ \t]+|[ \t]+$/, "", status)
  
  # Print the formatted line
  print timestamp " - " timestamp " BMC INFO: " name " | " status " | " value
}'

while true; do
  CURRENT_DATE=$(date +"%Y-%m-%d %H:%M:%S,%3N")
  BMC_READINGS=$(ipmitool sdr | grep -v 'ns' | egrep -i 'hgx|fan|inlet|temp|rpm')
  
  # Pass the awk script to awk with the timestamp variable
  echo "$BMC_READINGS" | awk -v timestamp="$CURRENT_DATE" "$AWK_SCRIPT" >> "$LOG_FILE"
  
  sleep "$INTERVAL"
done
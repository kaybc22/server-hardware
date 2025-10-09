#!/bin/bash



# Detect the latest matching files
bmclog=$(ls -t | grep -i "bmc_info" | head -n 1)
gputemp=$(ls -t | grep -i "nv_gpu" | head -n 1)

# Inject into Python script
echo "MONITORED_FILES = [\"$bmclog\", \"$gputemp\"]" >> monitor_web_v1.py

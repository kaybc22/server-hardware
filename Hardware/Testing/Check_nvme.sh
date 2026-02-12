#!/bin/bash

RED="\e[31m"
GREEN="\e[32m"
RESET="\e[0m"

echo "Checking NVMe SMART health..."
echo "-------------------------------------"

for dev in $(ls /dev/nvme?n?); do
    # Skip if device does not exist
    if [ ! -e "$dev" ]; then
        continue
    fi

    # Run smartctl and capture health line
    result=$(smartctl -a "$dev" 2>/dev/null | grep "SMART overall-health")

    if echo "$result" | grep -q "PASSED"; then
        echo -e "${GREEN}${dev}: PASSED${RESET}"
    elif echo "$result" | grep -q "FAILED"; then
        echo -e "${RED}${dev}: FAILED${RESET}"
    else
        echo -e "${RED}${dev}: UNKNOWN / NO DATA${RESET}"
    fi
done

echo "-------------------------------------"
echo "SMART scan complete."

#!/bin/bash

RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
RESET="\e[0m"

FAILED_DEVICES=()

echo "Checking NVMe SMART health..."
echo "-------------------------------------"

for dev in /dev/nvme?n?; do
    # Skip if device does not exist
    [ -e "$dev" ] || continue

    result=$(smartctl -H "$dev" 2>/dev/null | grep -i "SMART overall-health")

    if echo "$result" | grep -q "PASSED"; then
        echo -e "${GREEN}${dev}: PASSED${RESET}"
    elif echo "$result" | grep -q "FAILED"; then
        echo -e "${RED}${dev}: FAILED${RESET}"
        FAILED_DEVICES+=("$dev")
    else
        echo -e "${YELLOW}${dev}: UNKNOWN / NO DATA${RESET}"
    fi
done

echo "-------------------------------------"

# If there are failed devices
if [ ${#FAILED_DEVICES[@]} -gt 0 ]; then
    echo -e "${RED}The following devices FAILED:${RESET}"
    for dev in "${FAILED_DEVICES[@]}"; do
        echo "  $dev"
    done

    echo ""
    read -p "Do you want to run nvme format on FAILED devices? (yes/no): " confirm

    if [[ "$confirm" == "yes" ]]; then
        for dev in "${FAILED_DEVICES[@]}"; do
            echo -e "${YELLOW}Formatting $dev ...${RESET}"
            sudo nvme format "$dev" --force
        done
        echo -e "${GREEN}Formatting completed.${RESET}"
    else
        echo "Format skipped."
    fi
else
    echo -e "${GREEN}All devices PASSED. No formatting needed.${RESET}"
fi

echo "SMART scan complete."


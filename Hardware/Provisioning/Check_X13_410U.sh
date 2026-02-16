#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 IP1 IP2 ..."
  exit 1
fi

for ip in "$@"; do
  echo "Checking $ip ..."

  #ipmitool
  ipmitool -I lan -H "$ip" -U ADMIN -P ADMIN i2c bus=2 0xe2 0 0x50
  ipmitool -I lan -H "$ip" -U ADMIN -P ADMIN i2c bus=2 0xa8 0 0x0

  # Read only the FIRST line of output
  result=$(ipmitool -I lan -H "$ip" -U ADMIN -P ADMIN i2c bus=2 0xa0 1 0x2 | head -n 1 | tr -d '[:space:]')

  echo "Raw return: $result"

  case "$result" in
    "00")
      echo -e "\033[32m--> This is a 4U system (LC)\033[0m" 
      ;;
    "01")
      echo -e "\033[32m--> This is a 10U system (AC)\033[0m"
      ;;
    *)
      echo -e "\033[31m--> Unknown return value\033[0m"
      ;;
  esac

  echo
done

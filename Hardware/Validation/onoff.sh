#!/bin/bash

LOGS="$1"
ADMIN
ADMIN

if [ -z "$LOGS" ]; then
  echo "Log location is empty"
fi

remote (){
for i in {1..40}; do
  echo "Power cycling the system: Attempt $i"
  ipmitool -I lanplus -H 172.31.35.119 -U $(ADMIN) -P $(ADMIN) chassis power cycle
  date >> timestamp.log
  echo "Waiting for 10 minutes..."
  #lspci        > pci_$i.log
  sleep 720 # Sleep for 10 minutes
done
}

local (){
#crontab -e
#*/15 * * * * /path/to/your/script/a.txt

for i in {1..40}; do
  echo "Power cycling the system: Attempt $i"
  ipmitool chassis power cycle
  date >> $LOGS/onoff_timestamp.txt
  echo "Waiting for 10 minutes..."
  sleep 720 # Sleep for 10 minutes
done
}

#local
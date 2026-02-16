#!/bin/bash
USAGE="Usage: '$0 -H [ip] to slow fan speed"
IP=""


echo "Slowing fan speed periodically..."
#Enable SDHM mode for manual fan speed contol
#ipmitool raw 0x30 0x70 0x66 2 1
#Disable SDHM mode and go back to sensor automatic control
#ipmitool raw 0x30 0x70 0x66 2 0

#ipmitool -H <IP> -U ADMIN -P ADMIN raw 0x30 0x70 0x66 SET/GET Zone Value (in HEX/DEC)
#Set zone 1 to 100% duty cycle
#ipmitool -H <IP> -U ADMIN -P ADMIN raw 0x30 0x70 0x66 1 0 100
#Get zone 0 current duty cycle
#ipmitool -H <IP> -U ADMIN -P ADMIN raw 0x30 0x70 0x66 0 0

while [ 1 ]; do
    `ipmitool -H $1 -U ADMIN -P ADMIN raw 0x30 0x70 0x66 1 0 0x20`
    `ipmitool -H $1 -U ADMIN -P ADMIN raw 0x30 0x70 0x66 1 1 0x20`
    `ipmitool -H $1 -U ADMIN -P ADMIN raw 0x30 0x70 0x66 1 2 0x20`
    `ipmitool -H $1 -U ADMIN -P ADMIN raw 0x30 0x70 0x66 1 3 0x20`
    echo "Slowing fan speed now..."
    sleep 10
done  
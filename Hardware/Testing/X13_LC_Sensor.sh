#!/bin/bash


BMCIP=$1
BMCLOGIN=$2
BMCPASS=$3
IPMI_CMD_PREFIX="ipmitool -H $BMCIP -U $BMCLOGIN -P $BMCPASS"

if [ -z "$BMCIP" ] || [ -z "$BMCLOGIN" ] || [ -z "$BMCPASS" ]; then
  echo "Usage: $0 <bmc-IP> <username> <password>"
  exit 1
fi


function hex2volt {

MSB_HEX=$1
LSB_HEX=$2

# Convert hex to decimal
MSB_DEC=$(( 0x$MSB_HEX ))
LSB_DEC=$(( 0x$LSB_HEX ))

# Combine to 10-bit value
RAW=$(( (MSB_DEC << 2) | (LSB_DEC >> 6) ))

# Calculate voltage and determine status
awk -v msb="$MSB_HEX" -v lsb="$LSB_HEX" -v raw="$RAW" 'BEGIN {
  voltage = raw * 0.002
  if (voltage > 1.94) {
    status = "            ABSENT"
  } else if (voltage < 1.5) {
    status = "                      LEAKAGE"
  } else {
    status = "Enabled"
  }
  printf "MSB: %s, LSB: %s -> Voltage: %.3f V,  Status: %s\n", msb, lsb, voltage, status
}'
}


for j in {1..2}; do
   echo "--- $j, $BMCIP -------------------------------------------------------------------"
   date
   echo "[1] PCIE Switch Cold Plate"
   vol50aH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0a`
   vol50aL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0f`
   hex2volt $vol50aH $vol50aL
   echo ""
   echo "[2] MEM Cold Plate"
   vol50cH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0c`
   vol50cL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0f`
   hex2volt $vol50cH $vol50cL
   echo ""
   echo "[3] CPU1 Cold Plate"
   vol50dH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0d`
   vol50dL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0f`
   hex2volt $vol50dH $vol50dL
   echo ""
   echo "[4] CPU2 Cold Plate"
   vol50eH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0e`
   vol50eL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0f`
   hex2volt $vol50eH $vol50eL
   echo ""
   echo "[5] GPU Cold Plate 4"
   vol5EaH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0a`
   vol5EaL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0f`
   hex2volt $vol5EaH $vol5EaL
   echo ""
   echo "[6] GPU Cold Plate 1"
   vol5EcH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0c`
   vol5EcL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0f`
   hex2volt $vol5EcH $vol5EcL
   echo ""
   echo "[7] GPU Cold Plate 2"
   vol5EdH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0d`
   vol5EdL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0f`
   hex2volt $vol5EdH $vol5EdL
   echo ""
   echo "[8] GPU Cold Plate 3"
   vol5EeH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0e`
   vol5EeL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x5e 0x01 0x0f`
   hex2volt $vol5EeH $vol5EeL
   echo ""
   echo "????????????"
   vol5EeH=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0a`
   vol5EeL=`$IPMI_CMD_PREFIX raw 0x06 0x52 0x0f 0x50 0x01 0x0f`
   hex2volt $vol5EeH $vol5EeL  
   sleep 5
done
echo "--- DONE ---"

#Appendix:
#[1] PCIE Switch Cold Plate
#[2] MEM Cold Plate
#[3] CPU1 Cold Plate
#[4] CPU2 Cold Plate
#[5] GPU Cold Plate 4
#[6] GPU Cold Plate 1
#[7] GPU Cold Plate 2
#[8] GPU Cold Plate 3




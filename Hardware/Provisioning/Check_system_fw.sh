#!/bin/bash
# Usage: ./Check_system_fw.sh <bmc-url> <username> <password>
#./Check_system_fw.sh https://172.31.53.157/ ADMIN ADMIN
 

BMC_URL="$1"
USERNAME="$2"
PASSWORD="$3"

# ANSI color code for green
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ -z "$BMC_URL" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
  echo "Usage: $0 <bmc-url> <username> <password>"
  exit 1
fi


# Fetch inventory and extract member URIs
MEMBERS=$(curl -skL -u "$USERNAME:$PASSWORD" -X GET "$BMC_URL/redfish/v1/UpdateService/FirmwareInventory/" | jq -r '.Members[]."@odata.id"')

# Print header
printf "%-30s %-20s %s\n" "Id" "Version" "URI"
echo "------------------------------------------------------------"

# Fetch details for each component
for URI in $MEMBERS; do
  FULL_URI="$BMC_URL$URI"
  DETAILS=$(curl -skL -u "$USERNAME:$PASSWORD" -X GET "$FULL_URI" 2>/dev/null | jq -r '.Id + "|" + (.Version // "N/A") + "|" + "'"$URI"'"')
  if [ "$DETAILS" != "null|N/A|$URI" ]; then
    # Split DETAILS into Id, Version, URI
    ID=$(echo "$DETAILS" | cut -d'|' -f1)
    VERSION=$(echo "$DETAILS" | cut -d'|' -f2)
    URI_FIELD=$(echo "$DETAILS" | cut -d'|' -f3)
    # Print with Version in green
    printf "%-30s ${GREEN}%-20s${NC} %s\n" "$ID" "$VERSION" "$URI_FIELD"
  else
    echo "ERROR fetching $URI"
  fi
done

#GPU FW Update
#if [ -z "$1" ]; then
#  echo "Usage: $0 Path to the GPU FW PATH_FW"
#  exit 1
#fi
#@fw.fwpkg
#FW=$1
#curl -X POST BMC_URL/redfish/v1/UpdateService/upload -H 'content-type: multipart/form-data' -F UpdateFile=@FW -F 'UpdateParameters={"Targets":[""], "@Redfish.OperationApplyTime": "OnReset"}' -u "$USERNAME:$PASSWORD" -k
#curl -X POST -T nvfw_HGX-B100-B200x8_0009_241104.1.2_a_custom_prod-signed.fwpkg -kv -u "user:passwd" BMC_URL/redfish/v1/Oem/Supermicro/HGX_B200/UpdateService/
-X POST -H "Content-Type: application/json" --data "{ \"ImageURI\" : \"http://%SERVER_IP%/Nvidia.fwpkg\" , \"TransferProtocol\" : \"HTTP\", [\"/redfish/v1/UpdateService/FirmwareInventory/HGX_FW_BMC_0\"]"
redfish/v1/UpdateService/FirmwareInventory/HGX_FW_PCIeRetimer_0



 
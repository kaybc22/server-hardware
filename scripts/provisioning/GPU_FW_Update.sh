#!/bin/bash
# Usage: ./GPU_FW_update.sh <bmc-url> <username> <password> <Path-To-GPU FW>
#./GPU_FW_update.sh 172.31.53.157 ADMIN ADMIN /opt/testing


BMC_IP="$1"
USERNAME="$2"
PASSWORD="$3"
FWPKG="$4"

update(){
curl https://$BMC_IP/redfish/v1/UpdateService/upload \
  -H 'Content-Type: multipart/form-data' \
  -F UpdateFile=@$FWPKG \
  -F 'UpdateParameters={\"Targets\":[\"\"], \"@Redfish.OperationApplyTime\": \"OnReset\"}' \
  -u "$USERNAME:$PASSWORD"
}


if [ -z "$BMC_IP" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ] || [ -z "$FWPKG" ]; then
  echo "Usage: $0 <bmc-ip> <username> <password> <Path-To-GPU FW>"
  exit 1
fi

echo "curl https://$BMC_IP/redfish/v1/UpdateService/upload \
  -H 'Content-Type: multipart/form-data' \
  -F UpdateFile=@$FWPKG \
  -F 'UpdateParameters={\"Targets\":[\"\"], \"@Redfish.OperationApplyTime\": \"OnReset\"}' \
  -u \"$USERNAME:$PASSWORD\""

#update
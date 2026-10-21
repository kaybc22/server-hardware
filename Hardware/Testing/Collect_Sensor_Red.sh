#!/usr/bin/env bash
# bmc_temp_collect.sh
# Collect DGX/HGX inlet and ambient temperatures from Redfish + IPMI SDR.
#
# Usage:
#   ./bmc_temp_collect.sh -H <bmc-ip> -U <user> -P <password>
#   BMC=10.0.0.20 U=admin P=secret ./bmc_temp_collect.sh
#   ./bmc_temp_collect.sh -H 10.0.0.20 -U admin -P secret --out /tmp/bmc-temp
#
# Password is accepted on the CLI or via env P / BMC_PASSWORD.
# It is not written to logs.

set -uo pipefail

BMC="${BMC:-}"
U="${U:-}"
P="${P:-${BMC_PASSWORD:-}}"
OUT=""
INSECURE=1

usage() {
    sed -n '2,14p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--bmc|--host) BMC="${2:?}"; shift 2 ;;
        -U|--user)       U="${2:?}"; shift 2 ;;
        -P|--password)   P="${2:?}"; shift 2 ;;
        --out|-o)        OUT="${2:?}"; shift 2 ;;
        --insecure)      INSECURE=1; shift ;;
        --secure)        INSECURE=0; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$BMC" || -z "$U" || -z "$P" ]]; then
    echo "Need BMC host, user, and password." >&2
    usage 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "[CRITICAL] curl not found." >&2
    exit 127
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "[CRITICAL] jq not found." >&2
    exit 127
fi

CURL=(curl -sS --max-time 20)
if [[ "$INSECURE" -eq 1 ]]; then
    CURL+=(-k)
fi
CURL+=(-u "$U:$P")

BASE="https://${BMC}"

if [[ -n "$OUT" ]]; then
    mkdir -p "$OUT"
    exec > >(tee "$OUT/bmc_temp_collect.txt")
fi

echo "=== BMC temperature collect ==="
echo "bmc  : $BMC"
echo "user : $U"
echo "time : $(date -Is)"
echo

rf_get() {
    local path="$1"
    "${CURL[@]}" "${BASE}${path}"
}

echo "--- Redfish sensor members (temp|inlet|ambient) ---"
if ! rf_get /redfish/v1/Chassis/DGX/Sensors \
    | jq -r '.Members[]?."@odata.id" // empty' \
    | grep -iE 'temp|inlet|ambient'; then
    echo "[WARN] Chassis/DGX/Sensors list failed or no matching members"
fi
echo

echo "--- TEMP_AMBIENT ---"
rf_get /redfish/v1/Chassis/DGX/Sensors/TEMP_AMBIENT \
    | jq '{Name,Reading,ReadingUnits,Status,Thresholds,PhysicalContext}' \
    || echo "[WARN] TEMP_AMBIENT not available"
echo

echo "--- TEMP_MB_INLET ---"
rf_get /redfish/v1/Chassis/DGX/Sensors/TEMP_MB_INLET \
    | jq '{Name,Reading,Status,Thresholds}' \
    || echo "[WARN] TEMP_MB_INLET not available"
echo

echo "--- HGX_Chassis_0_Inlet_0_Temp_0 ---"
rf_get /redfish/v1/Chassis/HGX_Chassis_0/Sensors/HGX_Chassis_0_Inlet_0_Temp_0 \
    | jq '{Name,Reading,Status,Thresholds}' \
    || echo "[WARN] HGX inlet sensor not available"
echo

echo "--- ThermalSubsystem TemperatureReadingsCelsius ---"
rf_get /redfish/v1/Chassis/DGX/ThermalSubsystem/ThermalMetrics \
    | jq '.TemperatureReadingsCelsius[]? | {DeviceName,Reading}' \
    || echo "[WARN] ThermalMetrics not available"
echo

echo "--- ipmitool sdr type Temperature ---"
if command -v ipmitool >/dev/null 2>&1; then
    ipmitool -I lanplus -H "$BMC" -U "$U" -P "$P" sdr type Temperature \
        || echo "[WARN] ipmitool SDR Temperature failed"
else
    echo "[WARN] ipmitool not installed — skip SDR"
fi

echo
echo "=== done ==="


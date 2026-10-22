
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# HGX Redfish Hardware Inventory
# Usage:
#   ./hgx_inventory.sh https://BMC_IP username
#   BMC_PASSWORD='secret' ./hgx_inventory.sh https://BMC_IP username
#   ./hgx_inventory.sh https://BMC_IP username 'password'
# ============================================================

BMC_URL="${1:-}"
USERNAME="${2:-}"
PASSWORD="${3:-}"

# Set to 1 only if the BMC uses a self-signed/untrusted certificate.
INSECURE_TLS="${INSECURE_TLS:-1}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
MAX_TIME="${MAX_TIME:-20}"

if [[ -z "$BMC_URL" || -z "$USERNAME" ]]; then
    echo "Usage: $0 <bmc-url> <username> [password]"
    echo "Example: $0 https://192.168.1.100 root"
    exit 1
fi

# Prompt rather than require a password in shell history.
if [[ -z "$PASSWORD" ]]; then
    PASSWORD="${BMC_PASSWORD:-}"
fi

if [[ -z "$PASSWORD" ]]; then
    read -rsp "BMC password: " PASSWORD
    echo
fi

# Normalize the URL.
BMC_URL="${BMC_URL%/}"

# Basic dependency checks.
for tool in curl jq; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: Required command '$tool' is not installed." >&2
        exit 1
    fi
done

CURL_TLS=()
if [[ "$INSECURE_TLS" == "1" ]]; then
    CURL_TLS+=(--insecure)
fi

# GET a Redfish resource.
# Return JSON on stdout; errors go to stderr.
redfish_get() {
    local endpoint="$1"
    local response
    local http_code
    local url="${BMC_URL}${endpoint}"

    if ! response="$(
        curl "${CURL_TLS[@]}" \
            --silent --show-error \
            --connect-timeout "$CONNECT_TIMEOUT" \
            --max-time "$MAX_TIME" \
            --user "$USERNAME:$PASSWORD" \
            --header "Accept: application/json" \
            --output /dev/stderr \
            --write-out '%{http_code}' \
            --request GET \
            "$url" 2>/dev/null
    )"; then
        echo "ERROR: Connection failed: $url" >&2
        return 1
    fi

    http_code="$response"

    if [[ ! "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        echo "ERROR: HTTP $http_code from $url" >&2
        return 1
    fi
}

# Simpler GET implementation that captures the response body
# and HTTP status separately.
redfish_get_json() {
    local endpoint="$1"
    local url="${BMC_URL}${endpoint}"
    local tmp_file http_code

    tmp_file="$(mktemp)"
    if ! http_code="$(
        curl "${CURL_TLS[@]}" \
            --silent --show-error \
            --connect-timeout "$CONNECT_TIMEOUT" \
            --max-time "$MAX_TIME" \
            --user "$USERNAME:$PASSWORD" \
            --header "Accept: application/json" \
            --output "$tmp_file" \
            --write-out '%{http_code}' \
            --request GET "$url"
    )"; then
        echo "ERROR: Request failed: $url" >&2
        rm -f "$tmp_file"
        return 1
    fi

    if [[ ! "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        echo "ERROR: HTTP $http_code from $url" >&2
        cat "$tmp_file" >&2
        rm -f "$tmp_file"
        return 1
    fi

    if ! jq empty "$tmp_file" >/dev/null 2>&1; then
        echo "ERROR: Invalid JSON returned by $url" >&2
        rm -f "$tmp_file"
        return 1
    fi

    cat "$tmp_file"
    rm -f "$tmp_file"
}

# Print common hardware identity fields from a JSON object.
print_inventory() {
    local endpoint="$1"
    local label="$2"
    local json

    echo
    echo "============================================================"
    echo "$label"
    echo "Endpoint: $endpoint"
    echo "============================================================"

    if ! json="$(redfish_get_json "$endpoint")"; then
        echo "Status: FAILED"
        return 0
    fi

    echo "$json" | jq -r '
        def fields:
            {
                Name:         (.Name // null),
                Model:        (.Model // null),
                Manufacturer: (.Manufacturer // null),
                PartNumber:   (.PartNumber // null),
                SerialNumber: (.SerialNumber // null),
                PowerState:   (.PowerState // null),
                Status:       (.Status.State // null),
                Health:       (.Status.Health // null)
            };

        if ([.Name, .Model, .PartNumber, .SerialNumber,
             .PowerState, .Manufacturer] | any(. != null))
        then
            fields
            | to_entries[]
            | select(.value != null)
            | "\(.key): \(.value)"
        else
            "No common inventory fields at the resource root."
        end
    '

    # Assembly resources may contain nested components.
    echo "$json" | jq -r '
        .. | objects
        | select(
            (.SerialNumber? != null) or
            (.PartNumber? != null) or
            (.Model? != null)
        )
        | select(
            .["@odata.id"]? != null or
            .Name? != null or
            .SerialNumber? != null
        )
        | [
            ("Name=" + (.Name // "N/A")),
            ("Model=" + (.Model // "N/A")),
            ("PartNumber=" + (.PartNumber // "N/A")),
            ("SerialNumber=" + (.SerialNumber // "N/A"))
          ]
        | join(" | ")
    ' | sort -u

    # Print the Redfish error/message if the resource contains one.
    echo "$json" | jq -r '
        .error? // empty
        | if type == "object" then
            .["@Message.ExtendedInfo"][]?.Message // empty
          else . end
    ' 2>/dev/null || true
}

# ------------------------------------------------------------
# Main inventory collection
# ------------------------------------------------------------

echo "HGX Redfish Inventory"
echo "BMC: $BMC_URL"
echo "User: $USERNAME"
echo "TLS certificate verification disabled: $INSECURE_TLS"

# Check the Redfish service root first.
print_inventory "/redfish/v1/" "Redfish Service Root"

# Chassis assemblies.
print_inventory \
    "/redfish/v1/Chassis/HGX_Chassis_0/Assembly" \
    "HGX Chassis Assembly"

print_inventory \
    "/redfish/v1/Chassis/HGX_BMC_0/Assembly" \
    "HGX BMC Assembly"

# Eight HGX GPU/SXM modules.
for i in {1..8}; do
    print_inventory \
        "/redfish/v1/Chassis/HGX_GPU_SXM_${i}/" \
        "GPU / SXM Module $i"
done

echo
echo "Inventory collection completed."

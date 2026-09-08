#!/bin/bash
# nvlink_health_check.sh
# Parse NVIDIA NVLink error counters (NVLink 5 style).
# Usage:
#   ./nvlink_health_check.sh                  # run nvidia-smi live
#   ./nvlink_health_check.sh /path/to/log.txt # parse existing log/file
#
# Hard-error counters (CRITICAL if > 0):
#   Malformed packet Errors, Buffer overrun Errors, Rx Errors,
#   Rx remote Errors, Rx General Errors, Local link integrity Errors,
#   Effective Errors, Symbol Errors, Link recovery failed events,
#   PLR Xmit Retry Blocks
#
# BER (scientific notation):
#   Effective BER / Symbol BER
#   15e-255 (underflows to ~0) is NORMAL
#   15e-3, 15e-1, or any BER > 1e-12 is CRITICAL
#
# Raw BER / Raw Errors:
#   Small integers seen on healthy links (hundreds / millions) are printed only
#   Extremely large values are CRITICAL
#
# FEC:
#   FEC Errors - 0 is the "good symbols" bucket — do not alert
#   FEC Errors - 1..15 CRITICAL when the count is high (or any count for 3+)

set -uo pipefail

CMD="/usr/bin/nvidia-smi nvlink --errorcounters"
TMP_ERR=$(mktemp /tmp/nvlink_smi.XXXXXX.err)
TMP_OUT=$(mktemp /tmp/nvlink_smi.XXXXXX.out)
trap 'rm -f "$TMP_ERR" "$TMP_OUT"' EXIT

if [[ $# -ge 1 && -f "$1" && -r "$1" ]]; then
    echo "=== Parsing log file: $1 ==="
    cp "$1" "$TMP_OUT"
else
    if [[ $# -ge 1 ]]; then
        echo "WARNING: '$1' is not a readable file — falling back to live nvidia-smi" >&2
    fi
    SMI=$(command -v nvidia-smi || true)
    if [[ -z "$SMI" && -x /usr/bin/nvidia-smi ]]; then
        SMI=/usr/bin/nvidia-smi
    fi
    if [[ -z "$SMI" ]]; then
        echo "[CRITICAL] nvidia-smi utility is not installed or not in PATH." >&2
        echo "Check nvidia-smi utility or the fabric-manager working properly." >&2
        echo "  which nvidia-smi" >&2
        echo "  ls -l /usr/bin/nvidia-smi" >&2
        echo "  systemctl status nvidia-fabricmanager" >&2
        exit 1
    fi

    # Pre-check: can nvidia-smi talk to the driver at all?
    set +e
    "$SMI" -L >"$TMP_OUT" 2>"$TMP_ERR"
    pre_rc=$?
    set -e
    pre_msg=$(cat "$TMP_OUT" "$TMP_ERR" 2>/dev/null || true)
    if [[ $pre_rc -ne 0 ]] || echo "$pre_msg" | grep -qiE "couldn't communicate with the NVIDIA driver|NVIDIA-SMI has failed|No devices were found|Driver/library version mismatch"; then
        echo "[CRITICAL] nvidia-smi cannot communicate with the NVIDIA driver." >&2
        echo "Check nvidia-smi utility or the fabric-manager working properly." >&2
        echo "" >&2
        echo "----- nvidia-smi output -----" >&2
        printf '%s\n' "$pre_msg" >&2
        echo "-----------------------------" >&2
        echo "" >&2
        echo "Next checks:" >&2
        echo "  $SMI -L" >&2
        echo "  lsmod | grep -E 'nvidia|nvidia_uvm|nvidia_drm'" >&2
        echo "  systemctl status nvidia-persistenced nvidia-fabricmanager" >&2
        echo "  journalctl -u nvidia-fabricmanager -n 50 --no-pager" >&2
        echo "  dmesg -T | grep -iE 'NVRM|Xid|NVLink|nvidia' | tail -50" >&2
        exit 1
    fi

    CMD="$SMI nvlink --errorcounters"
    echo "=== Running live: $CMD ==="
    set +e
    $CMD >"$TMP_OUT" 2>"$TMP_ERR"
    smi_rc=$?
    set -e
    combined=$(cat "$TMP_OUT" "$TMP_ERR" 2>/dev/null || true)
    if echo "$combined" | grep -qiE "couldn't communicate with the NVIDIA driver|NVIDIA-SMI has failed"; then
        echo "[CRITICAL] nvidia-smi nvlink failed: driver not reachable." >&2
        echo "Check nvidia-smi utility or the fabric-manager working properly." >&2
        printf '%s\n' "$combined" >&2
        exit 1
    fi
    if [[ $smi_rc -ne 0 ]]; then
        echo "WARNING: nvidia-smi exit=$smi_rc (continuing if stdout is non-empty)" >&2
        if [[ -s "$TMP_ERR" ]]; then
            echo "----- nvidia-smi stderr -----" >&2
            cat "$TMP_ERR" >&2
            echo "-----------------------------" >&2
        fi
    fi
    if [[ ! -s "$TMP_OUT" ]]; then
        echo "[CRITICAL] nvidia-smi produced no NVLink counters." >&2
        echo "Check nvidia-smi utility or the fabric-manager working properly." >&2
        echo "  $SMI nvlink -e" >&2
        echo "  $SMI nvlink --errorcounters -i 0" >&2
        echo "  systemctl status nvidia-fabricmanager" >&2
        exit 1
    fi
    echo "=== nvidia-smi lines: $(wc -l < "$TMP_OUT")  exit: $smi_rc ==="
fi

echo "=== NVLink health parse (CRITICAL only) ==="

set +e
cat "$TMP_OUT" | awk '
BEGIN {
    gpu = "N/A"
    link = "?"
    crit = 0
    shown = 0

    # BER worse than this is a real link issue (15e-255 underflows to 0)
    BER_CRIT = 1e-12

    # Raw BER printed as a large integer (healthy samples are hundreds–thousands)
    RAW_BER_CRIT = 1000000

    # Raw Errors: healthy links can show millions; huge lifetime dumps are bad
    RAW_ERR_CRIT = 1e12

    # FEC Errors - N (N>=1): high corrected-symbol counts
    FEC_HIGH = 1000000
    FEC_MULTIBIT = 1000
}

function trim(s) {
    sub(/^[[:space:]]+/, "", s)
    sub(/[[:space:]]+$/, "", s)
    return s
}

function num_of(s) {
    gsub(/,/, "", s)
    return s + 0
}

function emit(sev, line) {
    shown++
    if (sev == "CRITICAL") {
        crit++
        printf "[CRITICAL] GPU %s | Link %s | %s\n", gpu, link, line
    }
    # OK / zero / below-threshold lines are silent
}

/^GPU [0-9]+:/ {
    gpu = $2
    gsub(/:/, "", gpu)
    next
}

/Link [0-9]+:/ {
    link = $2
    gsub(/:/, "", link)
}

{
    line = trim($0)
    val = $NF
    gsub(/,/, "", val)
    n = num_of(val)
}

# --- Hard error counters: any count > 0 is CRITICAL ---
$0 ~ /Malformed packet Errors:|[[:space:]]Buffer overrun Errors:|[[:space:]]Rx Errors:|[[:space:]]Rx remote Errors:|[[:space:]]Rx General Errors:|[[:space:]]Local link integrity Errors:|[[:space:]]Effective Errors:|[[:space:]]Symbol Errors:|[[:space:]]Link recovery failed events:|[[:space:]]PLR Xmit Retry Blocks:/ {
    if ($0 ~ /BER:/) next
    if (n > 0) emit("CRITICAL", line)
    else       emit("OK", line)
    next
}

# --- Effective / Symbol BER (scientific notation) ---
# 15e-255 => n==0   NORMAL
# 15e-3   => 0.015  CRITICAL
# 15e-1   => 1.5    CRITICAL
$0 ~ /Effective BER:|[[:space:]]Symbol BER:/ {
    if (n > BER_CRIT) emit("CRITICAL", line)
    else              emit("OK", line)
    next
}

# --- Raw BER Lane 0 / 1 / Total (integer or sci) ---
$0 ~ /Raw BER Lane [01]:|[[:space:]]Raw BER Total:/ {
    # If it looks like scientific BER, use BER threshold; else integer threshold
    if (val ~ /[eE]/) {
        if (n > BER_CRIT) emit("CRITICAL", line)
        else              emit("OK", line)
    } else {
        if (n > RAW_BER_CRIT) emit("CRITICAL", line)
        else                  emit("OK", line)
    }
    next
}

# --- Raw Errors Lane * ---
$0 ~ /Raw Errors Lane / {
    if (n > RAW_ERR_CRIT) emit("CRITICAL", line)
    else                  emit("OK", line)
    next
}

# --- FEC Errors - N ---
$0 ~ /FEC Errors - / {
    fecn = 0
    # line like: Link 17: FEC Errors - 3: 175
    if (match(line, /FEC Errors - ([0-9]+):/)) {
        # POSIX: pull the number after "- "
        split(line, a, /FEC Errors - /)
        split(a[2], b, /:/)
        fecn = b[1] + 0
    }
    if (fecn == 0) {
        emit("OK", line)                    # good-symbol bucket, always large
    } else if (fecn >= 3 && n > FEC_MULTIBIT) {
        emit("CRITICAL", line)              # elevated multi-bit corrections
    } else if (n > FEC_HIGH) {
        emit("CRITICAL", line)              # FEC-1 / FEC-2 flood
    } else {
        emit("OK", line)
    }
    next
}

# --- Other recovery / PLR context (print, alert on non-zero retries already handled) ---
$0 ~ /Link recovery successful events:|[[:space:]]Total link recovery events:|[[:space:]]Tx discards:/ {
    if (n > 0) emit("CRITICAL", line)
    else       emit("OK", line)
    next
}

END {
    if (crit == 0)
        printf "=== OK | no CRITICAL NVLink counters (scanned %d matching fields) ===\n", shown
    else
        printf "=== Done | critical=%d ===\n", crit
    if (crit > 0) exit 2
}
'

status=${PIPESTATUS[1]}
if [[ $status -eq 2 ]]; then
    echo "RESULT: CRITICAL NVLink counters present" >&2
    exit 2
fi
exit 0

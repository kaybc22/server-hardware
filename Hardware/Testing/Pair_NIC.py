#!/usr/bin/env python3
import subprocess
import time
import re

# Configurable settings
TOTAL_NICS = 16
NICS = [f"mlx5_{i}" for i in range(TOTAL_NICS)]
DURATION = 1  # 1-second burst per test for rapid discovery
BASE_PORT = 18500

def get_mst_info():
    """Runs mst start and parses mst status -v to map mlx5_X to (Device Name, Bus ID)."""
    nic_map = {}
    try:
        # Start MST service
        subprocess.run(["mst", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Get verbose status
        res = subprocess.run(["mst", "status", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Line format example:
        # BlueField3(rev:1) /dev/mst/mt41692_pciconf0 48:00.0 mlx5_2 net-ibs1f0 ...
        # ConnectX8(rev:0)  /dev/mst/mt4131_pciconf7  ec:00.0 smi-mlx5_15,mlx5_15 net-ibp236s0 ...
        for line in res.stdout.splitlines():
            # Match: Device Name (e.g. BlueField3), Bus ID (e.g. 48:00.0 or ec:00.0), and target mlx5_X
            match = re.search(r'^\s*([A-Za-z0-9_\-]+)(?:\(rev:\d+\))?\s+.*?([0-9a-fA-F]{2,4}:[0-9a-fA-F]{2}\.[0-9a-fA-F])\s+.*?\b(mlx5_\d+)\b', line)
            if match:
                dev_name = match.group(1)
                bus_id = match.group(2)
                mlx_dev = match.group(3)
                nic_map[mlx_dev] = {
                    "device": dev_name,
                    "bus_id": bus_id
                }
    except Exception as e:
        print(f"[WARNING] Could not parse MST status: {e}")

    # Fallback default values if MST parsing fails or doesn't list a device
    for nic in NICS:
        if nic not in nic_map:
            nic_map[nic] = {"device": "Unknown", "bus_id": "N/A"}

    return nic_map

def parse_bandwidth(output):
    """Parses average bandwidth (Gb/sec) from ib_write_bw stdout."""
    match = re.search(r'^\s*\d+\s+\d+\s+[\d\.]+\s+([\d\.]+)\s+[\d\.]+', output, re.MULTILINE)
    if match:
        return float(match.group(1))
    return 0.0

def extract_error(output):
    """Extracts relevant RDMA / QP failure lines from log output."""
    error_patterns = [
        r"Failed to modify QP.*",
        r"Unable to Connect.*",
        r"Connection refused.*",
        r"Couldn't connect to.*",
        r"Operation timed out.*"
    ]
    
    found_errors = []
    for pattern in error_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        for m in matches:
            clean_msg = m.strip()
            if clean_msg not in found_errors:
                found_errors.append(clean_msg)

    if found_errors:
        return " | ".join(found_errors)
    return "0.00 Gb/s bandwidth (No physical RDMA traffic flow)"

def test_pair(server_nic, client_nic, port):
    """Executes ib_write_bw and verifies active bandwidth > 0 Gb/s."""
    server_cmd = [
        "ib_write_bw",
        f"--ib-dev={server_nic}",
        "--ib-port=1",
        f"--port={port}",
        f"--duration={DURATION}",
        "--report_gbits"
    ]
    
    client_cmd = [
        "ib_write_bw",
        f"--ib-dev={client_nic}",
        "--ib-port=1",
        f"--port={port}",
        f"--duration={DURATION}",
        "--report_gbits",
        "127.0.0.1"
    ]

    try:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(0.2)

        client_proc = subprocess.run(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=DURATION + 2
        )

        s_out, s_err = server_proc.communicate(timeout=2)
        c_out, c_err = client_proc.stdout or "", client_proc.stderr or ""
        
        combined_output = f"{c_out}\n{c_err}\n{s_out}\n{s_err}"

        error_triggers = ["Failed to modify QP", "Unable to Connect", "Connection refused"]
        has_error = any(err in combined_output for err in error_triggers)

        bw_val = parse_bandwidth(combined_output)

        if client_proc.returncode == 0 and not has_error and bw_val > 0.0:
            return True, f"{bw_val:.2f} Gb/s"
        else:
            err_msg = extract_error(combined_output)
            return False, err_msg

    except Exception as e:
        return False, str(e)
    finally:
        if 'server_proc' in locals() and server_proc.poll() is None:
            server_proc.kill()

def main():
    print("Initializing MST status and gathering NIC hardware info...")
    mst_info = get_mst_info()

    paired = {}
    available = set(NICS)

    print(f"Starting discovery across {TOTAL_NICS} NICs...\n")

    for i in range(len(NICS)):
        nic_a = NICS[i]
        if nic_a not in available:
            continue

        for j in range(i + 1, len(NICS)):
            nic_b = NICS[j]
            if nic_b not in available:
                continue

            port = BASE_PORT + (i * TOTAL_NICS + j)
            print(f"Testing {nic_a} ({mst_info[nic_a]['bus_id']}) <--> {nic_b} ({mst_info[nic_b]['bus_id']}) (port {port})... ", end="", flush=True)

            success, detail = test_pair(nic_a, nic_b, port)

            if success:
                print(f"[ CONNECTED ] -> Bandwidth: {detail}")
                paired[nic_a] = (nic_b, detail)
                paired[nic_b] = (nic_a, detail)
                available.remove(nic_a)
                available.remove(nic_b)
                break
            else:
                print(f"[ DISCONNECTED ] -> Reason: {detail}")

    # Output Final Summary
    print("\n" + "=" * 90)
    print("                            DISCOVERED NIC CONNECTIONS")
    print("=" * 90)
    print(f"  {'Device Name':<15} {'Bus ID':<10} {'NIC':<10} <=====>  {'NIC':<10} {'Bus ID':<10} {'Device Name':<15} | Bandwidth")
    print("-" * 90)
    
    seen = set()
    for nic, (peer, bw) in sorted(paired.items()):
        if nic not in seen:
            dev_a, bus_a = mst_info[nic]["device"], mst_info[nic]["bus_id"]
            dev_b, bus_b = mst_info[peer]["device"], mst_info[peer]["bus_id"]
            
            print(f"  {dev_a:<15} {bus_a:<10} {nic:<10} <=====>  {peer:<10} {bus_b:<10} {dev_b:<15} | {bw}")
            seen.add(nic)
            seen.add(peer)

    unpaired = set(NICS) - set(paired.keys())
    if unpaired:
        print("\nUNCONNECTED / DISCONNECTED NICS:")
        for nic in sorted(unpaired):
            dev, bus = mst_info[nic]["device"], mst_info[nic]["bus_id"]
            print(f"  {dev:<15} {bus:<10} {nic}")

if __name__ == "__main__":
    main()

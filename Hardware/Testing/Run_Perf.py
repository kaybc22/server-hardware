#!/usr/bin/env python3
import subprocess
import sys
import time
import re
from datetime import datetime

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

if len(sys.argv) < 3 or len(sys.argv[1:]) % 2 != 0:
    print("Usage: script.py mlx5_0 mlx5_1 [mlx5_2 mlx5_3 ...]")
    sys.exit(1)

devices = sys.argv[1:]
duration = 10
base_port = 18516
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"rdma_bw_test_{timestamp}.log"

pairs = [(devices[i], devices[i + 1]) for i in range(0, len(devices), 2)]

print("\nStarting RDMA bidirectional bandwidth tests...\n")
print(f"Log file: {logfile}\n")

results = []

with open(logfile, "w") as log:

    log.write(f"RDMA Bandwidth Test Log\n")
    log.write(f"Start Time: {datetime.now()}\n")
    log.write("====================================================\n\n")

    for idx, (dev1, dev2) in enumerate(pairs):
        port = base_port + idx

        console_msg = f"Running {dev1} <-> {dev2} (port {port})"
        print(console_msg)
        log.write(console_msg + "\n")

        cmd_server = [
            "ib_write_bw",
            f"--ib-dev={dev1}",
            "--ib-port=1",
            f"--port={port}",
            "--bidirectional",
            f"--duration={duration}",
            "--report_gbits"
        ]

        cmd_client = [
            "ib_write_bw",
            f"--ib-dev={dev2}",
            "--ib-port=1",
            f"--port={port}",
            "--bidirectional",
            f"--duration={duration}",
            "--report_gbits",
            "127.0.0.1"
        ]

        try:
            log.write(f"\nSERVER CMD: {' '.join(cmd_server)}\n")
            log.write(f"CLIENT CMD: {' '.join(cmd_client)}\n")

            server = subprocess.Popen(
                cmd_server,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            time.sleep(1)

            client = subprocess.Popen(
                cmd_client,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            client_out, client_err = client.communicate(timeout=duration + 10)
            server_out, server_err = server.communicate(timeout=5)

            # Log full raw output
            log.write("\n--- SERVER STDOUT ---\n")
            log.write(server_out or "")
            log.write("\n--- SERVER STDERR ---\n")
            log.write(server_err or "")

            log.write("\n--- CLIENT STDOUT ---\n")
            log.write(client_out or "")
            log.write("\n--- CLIENT STDERR ---\n")
            log.write(client_err or "")
            log.write("\n====================================================\n")

            # Extract average bandwidth
            match = re.findall(
                r"\d+\s+\d+\s+\S+\s+([\d.]+)\s+[\d.]+",
                client_out
            )

            if match:
                bw = match[-1]
                results.append((dev1, dev2, "PASS", bw))
                print(f"{GREEN}{dev1} <-> {dev2}: {bw} Gb/sec{RESET}")
            else:
                results.append((dev1, dev2, "FAIL", "N/A"))
                print(f"{RED}{dev1} <-> {dev2}: FAILED (no bandwidth data){RESET}")

        except Exception as e:
            log.write(f"\nERROR: {e}\n")
            log.write("====================================================\n")
            results.append((dev1, dev2, "FAIL", "ERROR"))
            print(f"{RED}{dev1} <-> {dev2}: FAILED ({e}){RESET}")

    # Final Summary in Log
    log.write("\n\n================= FINAL SUMMARY =================\n")
    for d1, d2, status, bw in results:
        log.write(f"{d1} <-> {d2}: {status} {bw}\n")
    log.write("=================================================\n")
    log.write(f"End Time: {datetime.now()}\n")

# Console Summary
print("\n================= TEST SUMMARY =================")
failed = False

for d1, d2, status, bw in results:
    if status == "PASS":
        print(f"{GREEN}{d1} <-> {d2}: {bw} Gb/sec{RESET}")
    else:
        print(f"{RED}{d1} <-> {d2}: FAILED{RESET}")
        failed = True

print("================================================\n")
print(f"Full debug log saved to: {logfile}\n")

sys.exit(1 if failed else 0)


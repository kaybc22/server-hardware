import subprocess
import requests
import json
import logging
import datetime
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# Your Redfish BMC IP address
REDFISH_IP = "172.31.32.165"

# Redfish username and password
# Highly recommended to use environment variables or a config file
USERNAME = os.getenv("REDFISH_USERNAME", "admin") # Fallback if env var not set
PASSWORD = os.getenv("REDFISH_PASSWORD", "Supermicro1234") # Fallback if env var not set

# Log file path
LOG_DIR = "/opt/testing/utls" # Ensure this directory exists or create it
LOG_FILE_NAME = "monitor_data.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Number of HGX GPUs to iterate through for Redfish
NUM_GPUS = 8

# Monitoring interval in seconds
MONITOR_INTERVAL_SECONDS = 3

# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'), # Append mode
        logging.StreamHandler(sys.stdout) # Also print to console
    ]
)
# Suppress InsecureRequestWarning from requests if using verify=False
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# --- Helper to run shell commands ---
def run_command(cmd_list):
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True, timeout=15)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(cmd_list)}' failed with error: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        logging.error(f"Command '{cmd_list[0]}' not found. Is it in your PATH?")
        return None
    except subprocess.TimeoutExpired:
        logging.error(f"Command '{' '.join(cmd_list)}' timed out.")
        return None

# --- Data Collection Functions ---

def get_nvidia_smi_data():
    """Collects and parses NVIDIA-SMI data."""
    # Mimics: nvidia-smi -q | egrep -iv "N/A|shutdown|max|slowdown|id" | egrep -i 'power draw|minor|bus|Current Temp|t.limit'
    cmd = ['nvidia-smi', '-q']
    output = run_command(cmd)
    if not output:
        return ["NVIDIA-SMI data collection failed."]

    lines = []
    relevant_keywords = ['Power Draw', 'Minor Number', 'Bus Id', 'Current Temp', 'Thermal.t_limit']
    for line in output.split('\n'):
        line_lower = line.lower()
        if any(keyword.lower() in line_lower for keyword in relevant_keywords):
            # Exclude lines containing specific negative keywords (similar to grep -iv)
            if not any(neg_keyword in line_lower for neg_keyword in ["n/a", "shutdown", "max", "slowdown", "id"]):
                lines.append(f"GPU SMI INFO: {line.strip()}")
    return lines

def get_bmc_info():
    """Collects and parses IPMItool SDR data."""
    # Mimics: ipmitool sdr | grep -v ns| egrep -i 'hgx|fan|inlet'
    cmd = ['ipmitool', 'sdr']
    output = run_command(cmd)
    if not output:
        return ["BMC data collection failed."]

    lines = []
    relevant_keywords = ['hgx', 'fan', 'inlet']
    for line in output.split('\n'):
        line_lower = line.lower()
        if "ns" not in line_lower: # grep -v ns
            if any(keyword.lower() in line_lower for keyword in relevant_keywords):
                parts = line.split('|')
                if len(parts) >= 3: # Ensure proper format
                    sensor_name = parts[0].strip()
                    value_unit = parts[1].strip()
                    status = parts[2].strip()
                    lines.append(f"BMC INFO: {sensor_name:<15} | {status:<2} | {value_unit}")
    return lines

def fetch_redfish_sensor(gpu_index, sensor_type):
    """Helper to fetch a single Redfish sensor and parse relevant data."""
    url_template = f"https://{REDFISH_IP}/redfish/v1/Chassis/HGX_GPU_SXM_{gpu_index}/Sensors/HGX_GPU_SXM_{gpu_index}_{sensor_type}"
    try:
        response = requests.get(url_template, auth=(USERNAME, PASSWORD), verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()

        if sensor_type == "TEMP_1":
            name = data.get("Name", "N/A")
            reading = data.get("Reading", "N/A")
            return f"RED_TEMP (GPU {gpu_index}) | Name: {name:<25} | Reading: {reading} {data.get('ReadingUnits', '')}"
        elif sensor_type == "Power_0":
            name = data.get("Name", "N/A")
            reading = data.get("Reading", "N/A")
            peak_reading = data.get("PeakReading", "N/A")
            if "time" not in name.lower() and "time" not in str(reading).lower(): # grep -iv time for power
                return f"RED_POWER (GPU {gpu_index}) | Name: {name:<25} | Current: {reading} {data.get('ReadingUnits', '')} | Peak: {peak_reading} {data.get('ReadingUnits', '')}"
    except requests.exceptions.RequestException as e:
        return f"RED_ERROR (GPU {gpu_index} {sensor_type}) | Failed to fetch: {e}"
    except json.JSONDecodeError:
        return f"RED_ERROR (GPU {gpu_index} {sensor_type}) | Invalid JSON response."
    return f"RED_ERROR (GPU {gpu_index} {sensor_type}) | Unknown error."

def get_all_gpu_redfish_info():
    """Collects Redfish data for all GPUs concurrently."""
    redfish_data_lines = []
    # Use ThreadPoolExecutor to make concurrent requests
    with ThreadPoolExecutor(max_workers=NUM_GPUS * 2) as executor: # 2 threads per GPU (temp + power)
        futures = []
        for i in range(1, NUM_GPUS + 1):
            futures.append(executor.submit(fetch_redfish_sensor, i, "TEMP_1"))
            futures.append(executor.submit(fetch_redfish_sensor, i, "Power_0"))

        for future in futures:
            result = future.result()
            if result:
                redfish_data_lines.append(f"GPU REDFISH INFO: {result}")
    return redfish_data_lines

# --- Main Monitoring Loop ---
def main():
    if USERNAME == "admin" and PASSWORD == "Supermicro1234":
        logging.warning("WARNING: Using default Redfish username/password. Please change them or set environment variables REDFISH_USERNAME/REDFISH_PASSWORD.")

    # Determine end time based on the original script's SECONDS logic
    # The original script doesn't define 'end', assuming it's an external variable.
    # For this Python script, we'll run indefinitely unless you define a duration here.
    # For example, to run for 1 hour (3600 seconds):
    # end_time_epoch = time.time() + 3600
    # Or to run indefinitely:
    # while True:

    # Run indefinitely for this example
    while True:
        snapshot_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"\n--- Snapshot at {snapshot_time} ---") # Header for each snapshot

        # 1. Collect NVIDIA-SMI info
        nvidia_smi_lines = get_nvidia_smi_data()
        for line in nvidia_smi_lines:
            logging.info(f"{snapshot_time} {line}")

        # 2. Collect BMC info
        bmc_lines = get_bmc_info()
        for line in bmc_lines:
            logging.info(f"{snapshot_time} {line}")

        # 3. Collect GPU Redfish info concurrently
        redfish_lines = get_all_gpu_redfish_info()
        for line in redfish_lines:
            logging.info(f"{snapshot_time} {line}")

        logging.info(f"--- Snapshot End ---\n")

        time.sleep(MONITOR_INTERVAL_SECONDS) # Wait for the next snapshot

if __name__ == "__main__":
    main()
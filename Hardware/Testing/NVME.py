import subprocess
import os
import re

def stress_nvme(device=None, duration=60):
    """
    Stress test NVMe device(s) using fio with random writes.
    If no device is specified, tests all available NVMe devices except the OS disk.
    
    Args:
        device (str, optional): Path to a specific NVMe device (e.g., /dev/nvme0n1)
        duration (int): Test duration in seconds
    
    Returns:
        bool: True if all tests complete successfully, False if any test fails
    """
    # Validate duration
    try:
        duration = int(duration)
        if duration <= 0:
            raise ValueError("Duration must be positive")
    except ValueError:
        print("Error: Invalid duration value")
        return False

    # Get OS disk to exclude it
    try:
        lsblk_output = subprocess.run(["lsblk", "-o", "NAME,MOUNTPOINT"], capture_output=True, text=True, check=True).stdout
        os_disk = None
        for line in lsblk_output.splitlines():
            if "/ " in line:  # Root filesystem mount point
                match = re.match(r"(\w+)", line)
                if match:
                    os_disk = match.group(1)
                    break
    except subprocess.CalledProcessError:
        print("Warning: Could not determine OS disk, proceeding without exclusion")
        os_disk = None

    # Determine devices to test
    devices = []
    if device:
        if os.path.exists(device):
            if os_disk and device.startswith(f"/dev/{os_disk}"):
                print(f"Error: {device} is the OS disk, cannot test")
                return False
            devices.append(device)
        else:
            print(f"Error: Device {device} does not exist")
            return False
    else:
        # Scan for NVMe devices (nvme0n1 to nvme9n1)
        for i in range(10):
            dev = f"/dev/nvme{i}n1"
            if os.path.exists(dev) and (not os_disk or not dev.startswith(f"/dev/{os_disk}")):
                devices.append(dev)
        if not devices:
            print("Error: No valid NVMe devices found")
            return False

    # Run fio test on each device
    success = True
    for dev in devices:
        print(f"Testing device: {dev}")
        fio_cmd = [
            "fio",
            "--name=nvme_test",
            f"--filename={dev}",
            "--rw=randwrite",
            "--bs=4k",
            "--size=1G",
            "--numjobs=4",
            "--time_based",
            f"--runtime={duration}",
            "--group_reporting"
        ]

        try:
            result = subprocess.run(fio_cmd, capture_output=True, text=True, check=True)
            print(f"NVMe stress test on {dev} completed successfully")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running fio on {dev}: {e.stderr}")
            success = False
        except Exception as e:
            print(f"Unexpected error on {dev}: {str(e)}")
            success = False

    return success

# Example usage
    # Test a specific device
    # stress_nvme(device="/dev/nvme0n1", duration=120)
    # Test all devices (excluding OS disk)
    # stress_nvme(duration=120)
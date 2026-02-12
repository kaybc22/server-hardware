import subprocess
import json

def get_mounted_devices():
    """Return a set of devices that are mounted (OS drive + others)."""
    mounts = set()
    result = subprocess.run(["lsblk", "-J", "-o", "NAME,MOUNTPOINT"], capture_output=True, text=True)
    data = json.loads(result.stdout)

    def walk(blocks):
        for blk in blocks:
            name = blk["name"]
            mount = blk.get("mountpoint")
            if mount:  # anything mounted counts as OS-related
                mounts.add(name)
            if "children" in blk:
                walk(blk["children"])

    walk(data["blockdevices"])
    return mounts


def get_all_nvme_devices():
    """Return a list of NVMe device names (e.g., nvme1n1)."""
    result = subprocess.run(["lsblk", "-dn", "-o", "NAME,TYPE"], capture_output=True, text=True)
    devices = []
    for line in result.stdout.splitlines():
        name, dev_type = line.split()
        if dev_type == "disk" and name.startswith("nvme"):
            devices.append(name)
    return devices


def run_smart_log(devices):
    """Run nvme smart-log on each device."""
    for dev in devices:
        print(f"\n=== SMART LOG for {dev} ===")
        cmd = ["nvme", "smart-log", f"/dev/{dev}"]
        subprocess.run(cmd)


if __name__ == "__main__":
    mounted = get_mounted_devices()
    all_nvme = get_all_nvme_devices()

    # Skip OS drive by excluding any NVMe device that has mounted partitions
    non_os_nvme = [dev for dev in all_nvme if dev not in mounted]

    print("Detected NVMe drives:", all_nvme)
    print("Mounted (OS-related) devices:", mounted)
    print("NVMe drives to check:", non_os_nvme)

    run_smart_log(non_os_nvme)


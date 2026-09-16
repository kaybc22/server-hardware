#!/bin/bash

#Drivers=$(ls /sys/bus/pci/drivers/)
# Actions: unbind, bind, remove, rescan, reset

DEVICE="$1"
ACTION="$2"

usage() {
    echo "Usage: $0 <PCI_BUS_ID> <action>"
    echo "Actions:"
    echo "  unbind   - Unbind device from its driver"
    echo "  bind     - Bind device to its driver"
    echo "  remove   - Remove device from PCIe tree"
    echo "  rescan   - Rescan PCIe bus"
    echo "  reset    - Reset PCIe device"
    echo ""
    echo "Example:"
    echo "  $0 0000:dc:00.0 unbind"
    exit 1
}

# Validate input
if [[ -z "$DEVICE" || -z "$ACTION" ]]; then
    usage
fi

SYSFS="/sys/bus/pci/devices/$DEVICE"

if [[ ! -d "$SYSFS" && "$ACTION" != "rescan" ]]; then
    echo "Error: PCI device $DEVICE not found under /sys/bus/pci/devices"
    exit 1
fi

case "$ACTION" in

    unbind)
        DRIVER=$(basename "$(readlink "$SYSFS/driver")")
        echo "Unbinding $DEVICE from driver $DRIVER"
        echo "$DEVICE" | sudo tee "/sys/bus/pci/drivers/$DRIVER/unbind"
        ;;

    bind)
        DRIVER=$(basename "$(readlink "$SYSFS/driver")" 2>/dev/null)

        if [[ -z "$DRIVER" ]]; then
            echo "Device is not currently bound. Attempting to detect driver..."
            DRIVER=$(lspci -k -s "$DEVICE" | awk '/Kernel driver in use/ {print $5}')
        fi

        if [[ -z "$DRIVER" ]]; then
            echo "Error: Could not detect driver for $DEVICE"
            exit 1
        fi

        echo "Binding $DEVICE to driver $DRIVER"
        echo "$DEVICE" | sudo tee "/sys/bus/pci/drivers/$DRIVER/bind"
        ;;

    remove)
        echo "Removing PCI device $DEVICE"
        echo 1 | sudo tee "$SYSFS/remove"
        ;;

    rescan)
        echo "Rescanning PCI bus"
        echo 1 | sudo tee /sys/bus/pci/rescan
        ;;

    reset)
        if [[ ! -f "$SYSFS/reset" ]]; then
            echo "Error: Device does not support PCIe reset"
            exit 1
        fi
        echo "Resetting PCI device $DEVICE"
        echo 1 | sudo tee "$SYSFS/reset"
        ;;

    *)
        echo "Invalid action: $ACTION"
        usage
        ;;
esac

echo "Done."


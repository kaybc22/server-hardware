#!/bin/bash

LOGDIR="/var/log/nvme_fio"
mkdir -p "$LOGDIR"

echo "Detecting OS device..."
OS_DEVICE=$(lsblk -no PKNAME $(findmnt -no SOURCE /boot))
echo "OS is on: $OS_DEVICE"

echo "Detecting all NVMe devices..."
ALL_NVME=$(lsblk -dno NAME,TYPE | awk '$2=="disk" && $1 ~ /^nvme/ {print $1}')

echo "All NVMe drives: $ALL_NVME"
echo "Excluding OS drive: $OS_DEVICE"

for DEV in $ALL_NVME; do
    if [[ "$DEV" == "$OS_DEVICE" ]]; then
        echo "Skipping OS drive: $DEV"
        continue
    fi

    LOGFILE="$LOGDIR/${DEV}_fio_$(date +%Y%m%d_%H%M%S).log"

    echo "Starting fio on /dev/$DEV (log: $LOGFILE)"

    taskset -c 0 sudo time fio \
        --rw=read \
        --runtime=100 \
        --time_based \
        --ioengine=libaio \
        --group_reporting \
        --exitall \
        --filename=/dev/$DEV \
        --name=/dev/$DEV \
        --bs=4096k \
        --numjobs=16 \
        --iodepth=16 \
        --size=1G \
        --loops=1 \
        --invalidate=1 \
        --randrepeat=1 \
        --direct=1 \
        --norandommap \
        > "$LOGFILE" 2>&1 &

done

echo "All fio jobs started in parallel. Waiting for completion..."
wait
echo "All tests completed."


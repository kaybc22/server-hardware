#!/bin/bash

# Duration in seconds (1 hour)
DURATION=100      # seconds


# Stress_ng function
run_tress_ng_test() {
#DURATION=60        # seconds
CPU_LOAD=70        # target CPU percentage
VM_WORKERS=2       # number of memory stress workers

# Get total memory in bytes
TOTAL_MEM_BYTES=$(grep MemTotal /proc/meminfo | awk '{print $2 * 1024}')
# Calculate 70% of total memory
TARGET_MEM_BYTES=$((TOTAL_MEM_BYTES * 70 / 100))
# Memory per worker
BYTES_PER_WORKER=$((TARGET_MEM_BYTES / VM_WORKERS))

echo "==========================================="
echo "?? Total System Memory: $((TOTAL_MEM_BYTES / 1024 / 1024)) MB"
echo "?? Using ~70%: $((TARGET_MEM_BYTES / 1024 / 1024)) MB total"
echo "??  Each stress worker: $((BYTES_PER_WORKER / 1024 / 1024)) MB"
echo "?? CPU Load Target: ${CPU_LOAD}%"
echo "?? Duration: ${DURATION}s"
echo "==========================================="
#--cpu-method: 
#matrix	ALU / FPU	Multiplies random matrices, stable for load control
#fft	FPU-heavy	Floating-point transforms, stresses vector units
#sha256	Integer / cache	Hashing computations, less floating-point load
#prime	Integer / memory	Searches for prime numbers
#copy	Memory bandwidth	Copies large buffers repeatedly

echo "===== Running stress-ng ====="

# Run stress-ng with CPU + memory stress
stress-ng --cpu "$(nproc)" --cpu-load "${CPU_LOAD}" --cpu-method matrix \
          --vm "${VM_WORKERS}" --vm-bytes "${BYTES_PER_WORKER}" \
          --timeout "${DURATION}s" --metrics-brief

  echo "===== Stress Test Complete ====="
}

# Stress test function
run_stress_tests() {
  echo "===== Running stressapptest ====="
  stressapptest -W -s $DURATION -M $(($(free -m | awk '/Mem:/ {print int($2 * 0.8)}'))) -m 8
  echo "===== Stress Test Complete ====="
}

run_tress_ng_test
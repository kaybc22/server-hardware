#!/bin/bash
# =============================================================================
# Robust Memory Bandwidth Benchmark Script for HGX B200 / Xeon Systems
# Tests different DIMM population configurations
# Uses: STREAM, mbw, stress-ng
# =============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <config_name>"
    echo "Example: $0 8channel_full_2DPC"
    echo "         $0 4channel_half_1DPC"
    exit 1
fi

TEST_NAME="$1"
RESULTS_DIR="memory_bench_${TEST_NAME}_$(date +%Y%m%d_%H%M%S)"
THREADS=$(nproc)
RUNS=3

echo "====================================================="
echo "Memory Bandwidth Benchmark - Configuration: ${TEST_NAME}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "====================================================="

mkdir -p "${RESULTS_DIR}"

# ====================== System Information ======================
echo "Collecting system and memory information..."
{
    echo "=== System Info ==="
    lscpu | cat
    echo -e "\n=== NUMA Topology ==="
    numactl -H
    echo -e "\n=== Memory DIMM Details ==="
    sudo dmidecode --type memory | grep -E "Size:|Speed:|Locator:|Bank Locator|Manufacturer|Part Number" | head -n 40
    echo -e "\n=== Current Memory Usage ==="
    free -h
} > "${RESULTS_DIR}/00_system_info.txt"

# ====================== Install Dependencies ======================
echo "Installing required tools..."
sudo apt-get update -qq
sudo apt-get install -y build-essential numactl stress-ng mbw git

# ====================== Build STREAM (Fixed & Reliable) ======================
echo "Building STREAM benchmark..."

if [ ! -d "stream" ]; then
    git clone https://github.com/jeffhammond/STREAM.git stream
fi

cd stream
make -f Makefile.stream clean >/dev/null 2>&1 || true

# Best compilation method
echo "Compiling STREAM..."
gcc -O3 -fopenmp -march=native -ffast-math \
    -DSTREAM_ARRAY_SIZE=80000000 \
    stream.c -o stream_c.exe

# Fallback if above fails
if [ ! -f "stream_c.exe" ]; then
    echo "Primary compile failed, trying Makefile..."
    make -f Makefile.stream CC=gcc CFLAGS="-O3 -fopenmp -march=native" || true
fi

if [ ! -f "stream_c.exe" ]; then
    echo "ERROR: Failed to build STREAM executable!"
    exit 1
fi

cd ..

# ====================== Run Benchmarks ======================
echo "Running STREAM benchmark (${RUNS} runs)..."

for i in $(seq 1 $RUNS); do
    echo "→ STREAM Run $i/$RUNS ..."
    numactl --interleave=all ./stream/stream_c.exe -t $THREADS \
        > "${RESULTS_DIR}/stream_run${i}.txt" 2>&1
done

echo "Running mbw benchmark..."
mbw -t 5 -q 8192 2>&1 | tee "${RESULTS_DIR}/mbw.txt"

echo "Running stress-ng memory tests..."
stress-ng --matrix 0 --matrix-method all -t 20s --metrics-brief \
    > "${RESULTS_DIR}/stressng_matrix.txt" 2>&1

stress-ng --vm 0 --vm-bytes 70% -t 15s --metrics-brief \
    > "${RESULTS_DIR}/stressng_vm.txt" 2>&1

# ====================== Summary ======================
echo "====================================================="
echo "Benchmark completed for: ${TEST_NAME}"
echo "Results directory: ${RESULTS_DIR}"

echo -e "\n=== STREAM TRIAD Bandwidth Summary (Higher is Better) ==="
grep -E "Triad:" "${RESULTS_DIR}"/stream_run*.txt | tail -n 4

echo -e "\n=== mbw Summary ==="
tail -n 10 "${RESULTS_DIR}/mbw.txt"

echo -e "\nNext steps:"
echo "1. Change your DIMM population"
echo "2. Run again with a different name, e.g.:"
echo "   ./mem_bench.sh 4channel_half_1DPC"
echo "   ./mem_bench.sh 8channel_full_1DPC"

# Optional: Create a quick CSV summary
echo "Config,Triad_GB_s" > "${RESULTS_DIR}/summary.csv"
grep "Triad:" "${RESULTS_DIR}"/stream_run1.txt | awk '{print "'${TEST_NAME}'," $2}' >> "${RESULTS_DIR}/summary.csv" 2>/dev/null || true
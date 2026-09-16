#!/bin/bash
# =============================================================================
# mem_topo_map_v4.sh - Improved DIMM → NUMA → PCIe → CPU mapping
# =============================================================================

OUT="memory_topology_map_v4.csv"
echo "NUMA_Node,DIMM_Locator,Bank_Locator,Size_MB,CPU_Cores,Nearby_PCIe_Devices" > "$OUT"

echo "Collecting hardware topology..."

# 1. Collect DIMM information
declare -A dimm_bank dimm_size
while IFS= read -r line; do
    case "$line" in
        *"Bank Locator:"*) 
            bank=$(echo "$line" | awk -F': ' '{print $2}' | xargs) ;;
        *"Locator:"*) 
            locator=$(echo "$line" | awk -F': ' '{print $2}' | xargs) ;;
        *"Size:"*) 
            size=$(echo "$line" | awk -F': ' '{print $2}' | awk '{print $1}' | xargs)
            if [[ -n "$bank" && -n "$locator" && "$size" != "No" ]]; then
                dimm_bank["$locator"]="$bank"
                dimm_size["$locator"]="$size"
            fi
            ;;
    esac
done < <(sudo dmidecode -t memory 2>/dev/null)

# 2. Collect CPU cores per NUMA node
declare -A node_cores
while read -r line; do
    node=$(echo "$line" | grep -o 'NUMA node[0-9]*' | grep -o '[0-9]*')
    cores=$(echo "$line" | awk -F': ' '{print $2}' | tr -d '[:space:]')
    [[ -n "$node" && -n "$cores" ]] && node_cores["$node"]="$cores"
done < <(lscpu | grep -E "NUMA node[0-9]+ CPU")

# 3. Collect PCIe devices per NUMA node 
declare -A node_devs
for dev in /sys/bus/pci/devices/*; do
    [ -e "$dev/numa_node" ] || continue
    node=$(cat "$dev/numa_node" 2>/dev/null)
    [[ -z "$node" || "$node" -lt 0 ]] && node=0
    
    busid=$(basename "$dev")
    desc=$(lspci -s "$busid" 2>/dev/null | grep -E "NVIDIA|Mellanox|Ethernet|Infiniband|NVMe|RAID" | sed 's/.*://')
    
    if [[ -n "$desc" ]]; then
        node_devs[$node]="${node_devs[$node]}${busid}(${desc:0:25}); "
    fi
done

# 4. Match DIMMs to NUMA nodes 
for nid in /sys/devices/system/node/node*; do
    nodeid=$(basename "$nid" | grep -o '[0-9]\+')
    cpus="${node_cores[$nodeid]:-N/A}"
    
    for loc in "${!dimm_bank[@]}"; do
        bank="${dimm_bank[$loc]}"
        size="${dimm_size[$loc]}"
        
        # Improved matching logic
        if [[ "$bank" == *"$nodeid"* ]] || \
           [[ "$bank" == *"Node $nodeid"* ]] || \
           [[ "$bank" == *"CPU$nodeid"* ]] || \
           [[ "$bank" == *"CH$nodeid"* ]] || \
           [[ "$bank" == *"P$nodeid"* ]]; then
            
            nearby="${node_devs[$nodeid]:-None}"
            echo "$nodeid,$loc,$bank,${size} MB,\"$cpus\",\"${nearby}\""
            echo "$nodeid,$loc,$bank,${size} MB,\"$cpus\",\"${nearby}\"" >> "$OUT"
        fi
    done
done

echo "Done! Topology map saved to: $OUT"
echo "You can open it with Excel / LibreOffice."

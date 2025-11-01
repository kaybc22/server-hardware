#!/bin/bash
# --------------------------------------------------------------
# mem_topo_map_v3.sh
# DIMM â†’ NUMA â†’ PCIe â†’ CPU mapping
# --------------------------------------------------------------

OUT="memory_topology_map_v3.csv"
echo "NUMA_Node,DIMM_Locator,Bank_Locator,Size_MB,CPU_Cores,Nearby_Devices" > "$OUT"

# ðŸ§  1ï¸âƒ£ Collect DIMM information
declare -A dimm_bank dimm_size
while IFS= read -r line; do
  case "$line" in
    *"Bank Locator:"*) bank=$(echo "$line" | awk -F: '{print $2}' | xargs) ;;
    *"Locator:"*) locator=$(echo "$line" | awk -F: '{print $2}' | xargs) ;;
    *"Size:"*)
      size=$(echo "$line" | awk -F: '{print $2}' | xargs | awk '{print $1}')
      if [[ -n "$bank" && -n "$locator" ]]; then
        dimm_bank["$locator"]="$bank"
        dimm_size["$locator"]="$size"
      fi
      ;;
  esac
done < <(sudo dmidecode -t memory)

# ðŸ§® 2ï¸âƒ£ Collect CPU cores per NUMA node
declare -A node_cores
while read -r line; do
  node=$(echo "$line" | sed -E 's/.*NUMA node([0-9]+).*/\1/')
  cores=$(echo "$line" | sed -E 's/.*:\s*//' | tr -d '[:space:]')
  [ -n "$node" ] && [ -n "$cores" ] && node_cores["$node"]="$cores"
done < <(lscpu | grep -E "NUMA node[0-9]+ CPU")

# ðŸ” 3ï¸âƒ£ Collect PCI devices by NUMA node (GPU/NIC/NVMe)
declare -A node_devs
for devpath in /sys/bus/pci/devices/*; do
  [ -e "$devpath/numa_node" ] || continue
  node=$(cat "$devpath/numa_node" 2>/dev/null)
  [[ -z "$node" || "$node" -lt 0 ]] && node=0
  busid=$(basename "$devpath")
  desc=$(lspci -s "$busid" 2>/dev/null | grep -E "NVIDIA|Mellanox|Ethernet|NVMe" | sed 's/.*://')
  [ -n "$desc" ] && node_devs[$node]="${node_devs[$node]}${busid}-${desc}; "
done

# ðŸ§© 4ï¸âƒ£ Match DIMMs â†’ NUMA heuristic
for nid in /sys/devices/system/node/node*; do
  nodeid=$(basename "$nid" | grep -o '[0-9]\+')
  cpus="${node_cores[$nodeid]}"
  for loc in "${!dimm_bank[@]}"; do
    bank="${dimm_bank[$loc]}"
    size="${dimm_size[$loc]}"
    if [[ "$bank" == *"Node$nodeid"* ]]; then
      echo "$nodeid,$loc,$bank,$size MB,\"$cpus\",\"${node_devs[$nodeid]}\"" >> "$OUT"
    fi
  done
done

echo "âœ… Memory + CPU + PCI topology written to $OUT"



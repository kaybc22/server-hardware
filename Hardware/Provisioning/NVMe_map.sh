for p in /dev/disk/by-path/*nvme-*; do
  [[ -L "$p" && ! "$p" =~ -part ]] || continue
  dev=$(readlink -f "$p")
  info=$(nvme id-ctrl "$dev" 2>/dev/null)
  sn=$(echo "$info" | awk -F: '/sn /{gsub(/^ +| +$/,"",$2); print $2}')
  model=$(echo "$info" | awk -F: '/mn /{gsub(/^ +| +$/,"",$2); print $2}')
  used=$(nvme list | awk -v d="$dev" '$1==d {print $6, $7}')
  printf "%s -> %s  %s  %-40s  %s\n" \
    "$(basename "$p")" "$(basename "$dev")" "$sn" "$model" "$used"
done | sort -k3

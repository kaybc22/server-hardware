#!/bin/bash
# check_packages.sh - Check installed utilities and show their package info

# Example usage:
#   ./check_packages.sh sysinfo-snapshot.py stress-ng nvidia-smi

UTILITIES=("$@")

if [ ${#UTILITIES[@]} -eq 0 ]; then
  echo "Usage: $0 <utility1> <utility2> ..."
  exit 1
fi

echo "🔍 Checking utilities: ${UTILITIES[*]}"
echo "--------------------------------------------"

for util in "${UTILITIES[@]}"; do
  echo "➡️  Checking $util ..."
  
  # Check if command exists
  util_path=$(command -v "$util" 2>/dev/null)
  if [ -z "$util_path" ]; then
    echo "   ❌ $util not found in PATH"
    echo
    continue
  fi

  echo "   ✅ Found at: $util_path"

  # Find which package it belongs to
  pkg_name=$(dpkg -S "$util_path" 2>/dev/null | awk -F: '{print $1}' | head -n1)

  if [ -z "$pkg_name" ]; then
    echo "   ⚠️  No package found via dpkg (maybe manually installed or in container)"
    echo
    continue
  fi

  echo "   📦 Package: $pkg_name"

  # Show brief package info
  apt show "$pkg_name" 2>/dev/null | grep -E "Package:|Version:|Description:" | head -n3
  echo
done

echo "✅ Check completed."


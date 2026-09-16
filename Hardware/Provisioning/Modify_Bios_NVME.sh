#!/bin/bash
# =============================================
# Modify BIOS File for SAA to enable NVMe Driver Feature
# Input: IP(s) or full bios.* filenames
# Output: Creates current.file.$ip (original untouched)
# =============================================

echo "=== Enhanced NVMe Firmware Source Updater ==="
echo

# === Collect files to process ===
if [ $# -eq 0 ]; then
    echo "No arguments given → Processing ALL bios.* files"
    files=(bios.*)
else
    echo "Processing specified file(s)/IP(s): $@"
    files=()
    for arg in "$@"; do
        if [[ -f "$arg" ]]; then
            # User passed full filename (e.g. bios.10.58.141.1)
            files+=("$arg")
        elif [[ -f "bios.$arg" ]]; then
            # User passed only IP (e.g. 10.58.141.1)
            files+=("bios.$arg")
        else
            echo "⚠️  File not found: $arg (tried bios.$arg too)"
        fi
    done
fi

if [ ${#files[@]} -eq 0 ]; then
    echo "❌ No valid bios.* files found."
    exit 1
fi

count=0

for orig_file in "${files[@]}"; do
    # Extract IP part (remove "bios." prefix if present)
    ip="${orig_file#bios.}"
    
    new_file="current.file.${ip}"
    
    echo "📄 Processing: $orig_file  →  $new_file"
    
    # Step 1: Verify current setting
    if grep -q "NVMe Firmware Source" "$orig_file"; then
        grep "NVMe Firmware Source" "$orig_file"
        
        # Step 2: Backup original (keeps it 100% safe)
        cp -v "$orig_file" "${orig_file}.bak"
        
        # Step 3: Create NEW file with the change
        sed 's/selectedOption="Vendor Defined Firmware"/selectedOption="AMI Native Support"/g' \
            "$orig_file" > "$new_file"
        
        # Step 4: Verify the new file
        if grep -q 'selectedOption="AMI Native Support"' "$new_file"; then
            echo "✅ Successfully created $new_file with AMI Native Support"
        else
            echo "⚠️  Warning: Change could not be verified in new file"
        fi
    else
        echo "No 'NVMe Firmware Source' setting found in $orig_file"
    fi
    echo "----------------------------------------"
    ((count++))
done

echo "🎉 Done! Processed $count file(s)."
echo "   • Original files preserved (with .bak backups)"
echo "   • New modified files created as current.file.*"

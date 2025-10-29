#!/bin/bash

# --- Configuration ---
# The name of your log archive - now taken as the first argument
LOG_ARCHIVE="$1"
LOG_FILE_IN_ARCHIVE="run.log"         # The name of the log file inside the archive

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# --- Extraction Function ---
# This function handles the extraction of various archive types
extract_archive() {
    local archive_path="$1"
    local dest_dir="$2"
    local extension="${archive_path##*.}"
    local name_with_ext="${archive_path##*/}"

    # Use a case statement to determine the archive type and extract accordingly
    case "$name_with_ext" in
        *.tar.gz|*.tgz)
            echo "Detected .tar.gz or .tgz archive. Extracting..."
            tar -xzf "$archive_path" -C "$dest_dir"
            ;;
        *.tar.bz2|*.tbz2)
            echo "Detected .tar.bz2 or .tbz2 archive. Extracting..."
            tar -xjf "$archive_path" -C "$dest_dir"
            ;;
        *.tar.xz|*.txz)
            echo "Detected .tar.xz or .txz archive. Extracting..."
            tar -xJf "$archive_path" -C "$dest_dir"
            ;;
        *.tar)
            echo "Detected .tar archive. Extracting..."
            tar -xf "$archive_path" -C "$dest_dir"
            ;;
        *.zip)
            echo "Detected .zip archive. Extracting..."
            unzip -q "$archive_path" -d "$dest_dir"
            ;;
        *)
            echo "Error: Unsupported archive format for '$archive_path'."
            return 1
            ;;
    esac
    return $?
}

# --- Sub-Folder Analysis Function ---
analyze_subfolder() {
    local subfolder_path="$1"
    local activity_id="$2"

    if [ ! -d "$subfolder_path" ]; then
        echo "Skipping non-existent sub-folder: '$subfolder_path'."
        return 0
    fi

    echo "Analyzing sub-folder: $subfolder_path (for ID: $activity_id)"

    # Check for exception.log
    if [ -f "$subfolder_path/exception.log" ]; then
        echo -e "${RED}Issue detected: 'exception.log' found in '$subfolder_path'.${NC}"
    else
        echo -e "${GREEN}No 'exception.log' found in '$subfolder_path'.${NC}"
    fi

    # Search all *.log files for "Error Code" lines (excluding all-zero codes)
    ERROR_CODE_LINES=$(find "$subfolder_path" -type f -name "*.log" -print0 | xargs -0 grep -E "Error Code = [0-9A-F]{12}" 2>/dev/null | grep -vE "Error Code = 0{12}")
    if [ -n "$ERROR_CODE_LINES" ]; then
        echo -e "${RED}Error Code lines found in *.log files:${NC}"
        echo -e "${RED}${ERROR_CODE_LINES}${NC}"
    else
        echo -e "${GREEN}No 'Error Code = ' (non-zero) found in *.log files in '$subfolder_path'.${NC}"
    fi
}

# --- SXM Sub-Folder Analysis Function ---
analyze_sxm_subfolder() {
    local sxm_subfolder_path="$1"
    local activity_id="$2"

    if [ ! -d "$sxm_subfolder_path" ]; then
        echo "Skipping non-existent SXM sub-folder: '$sxm_subfolder_path'."
        return 0
    fi

    echo "Analyzing SXM sub-folder: $sxm_subfolder_path (for ID: $activity_id)"

    # Check for exception.log
    if [ -f "$sxm_subfolder_path/exception.log" ]; then
        echo -e "${RED}Issue detected: 'exception.log' found in '$sxm_subfolder_path'.${NC}"
    else
        echo -e "${GREEN}No 'exception.log' found in '$sxm_subfolder_path'.${NC}"
    fi

    # Search all *.log files for "Error Code" lines (excluding all-zero codes)
    ERROR_CODE_LINES=$(find "$sxm_subfolder_path" -type f -name "*.log" -print0 | xargs -0 grep -E "Error Code = [0-9A-F]{12}" 2>/dev/null | grep -vE "Error Code = 0{12}")
    if [ -n "$ERROR_CODE_LINES" ]; then
        echo -e "${RED}Error Code lines found in *.log files:${NC}"
        echo -e "${RED}${ERROR_CODE_LINES}${NC}"
    else
        echo -e "${GREEN}No 'Error Code = ' (non-zero) found in *.log files in '$sxm_subfolder_path'.${NC}"
    fi
}

# --- Script Logic ---

# Check if a log archive was provided as an argument
if [ -z "$LOG_ARCHIVE" ]; then
    echo "Usage: $0 <log_archive>"
    echo "Example: $0 logs-20250730-235531.tgz"
    echo "Example: $0 another-log-archive.zip"
    exit 1
fi

# Create a temporary directory for extraction
TEMP_DIR=$(mktemp -d -t log_extract_XXXXXX)
if [ $? -ne 0 ]; then
    echo "Error: Could not create temporary directory."
    exit 1
fi
echo "Created temporary directory: $TEMP_DIR"

# Ensure the temporary directory is cleaned up on exit
trap "echo 'Cleaning up temporary directory: $TEMP_DIR'; rm -rf $TEMP_DIR" EXIT

# Check if the log archive exists
if [ ! -f "$LOG_ARCHIVE" ]; then
    echo "Error: Log archive '$LOG_ARCHIVE' not found in the current directory."
    exit 1
fi

echo "Extracting '$LOG_ARCHIVE' to '$TEMP_DIR'..."
# Call the extraction function
if ! extract_archive "$LOG_ARCHIVE" "$TEMP_DIR"; then
    echo "Error: Failed to extract '$LOG_ARCHIVE'."
    exit 1
fi

# Construct the full path to run.log inside the extracted directory
RUN_LOG_PATH=$(find "$TEMP_DIR" -name "$LOG_FILE_IN_ARCHIVE" | head -n 1)

if [ -z "$RUN_LOG_PATH" ]; then
    echo "Error: '$LOG_FILE_IN_ARCHIVE' not found inside the extracted archive."
    exit 1
fi

echo "Found log file: $RUN_LOG_PATH"

# --- Early check for Final Result: PASS ---
if grep -q "Final Result: PASS" "$RUN_LOG_PATH"; then
    echo ""
    echo "--- Analysis Results for $LOG_FILE_IN_ARCHIVE ---"
    echo -e "${GREEN}Final Result: PASS found in log file. No further analysis required.${NC}"
    echo "-------------------------------------"
    exit 0
fi

# Determine the top-level directory where the archive was extracted
TOP_LEVEL_EXTRACTED_DIR=$(dirname "$RUN_LOG_PATH")

# Get all test lines that are NOT all zeros
ALL_TEST_RESULTS=$(grep -E "^(MODS|DGX)-[0-9A-F]{12}" "$RUN_LOG_PATH")
TRUE_ISSUES=$(echo "$ALL_TEST_RESULTS" | grep -vE "^(MODS|DGX)-0{12}")

# Count PASS and FAIL lines among true issues
PASS_COUNT=$(echo "$TRUE_ISSUES" | grep -E "\| OK$" | wc -l)
TOTAL_TRUE_ISSUES=$(echo "$TRUE_ISSUES" | wc -l)
FAIL_COUNT=$((TOTAL_TRUE_ISSUES - PASS_COUNT))

# --- Report Results ---
echo ""
echo "--- Analysis Results for $LOG_FILE_IN_ARCHIVE ---"
echo "Total Relevant Test Entries (True Issues): $TOTAL_TRUE_ISSUES"
echo "PASS Count: $PASS_COUNT"
echo "FAIL Count: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo -e "--- ${RED}Failed Test Entries (True Issues)${NC} ---"
    FAILED_TRUE_ISSUES_LINES=$(echo "$TRUE_ISSUES" | grep -vE "\| OK$")
    echo -e "${RED}${FAILED_TRUE_ISSUES_LINES}${NC}"

    echo ""
    echo "--- Detailed Error Codes from Failed Activities ---"
    echo "$FAILED_TRUE_ISSUES_LINES" | while IFS='|' read -r id activity_col rest_of_line; do
        # Extract and trim activity
        activity=$(echo "$activity_col" | xargs)

        # Skip if activity starts with "Ext"
        if [[ "$activity" == "Ext"* ]]; then
            echo "Skipping activity '$activity' (excluded by filter)."
            continue
        fi

        ERROR_FOLDER="$TOP_LEVEL_EXTRACTED_DIR/$activity"

        if [ -d "$ERROR_FOLDER" ]; then
            echo "Searching in folder: $ERROR_FOLDER (for ID: $(echo "$id" | xargs))"
            # Search for non-zero Error Codes in activity folder
            ERROR_CODE_LINES=$(find "$ERROR_FOLDER" -type f -print0 | xargs -0 grep -E "Error Code = [0-9A-F]{12}" 2>/dev/null | grep -vE "Error Code = 0{12}")
            if [ -n "$ERROR_CODE_LINES" ]; then
                echo -e "${RED}$ERROR_CODE_LINES${NC}"
            else
                echo "  No 'Error Code = ' (non-zero) found in files within '$ERROR_FOLDER'."
            fi

            # Call sub-folder analysis for this activity folder
            analyze_subfolder "$ERROR_FOLDER" "$(echo "$id" | xargs)"

            # Check for SXM sub-folders (SXM{1..8}_SN_[0-9]+) in parallel
            echo "Checking SXM sub-folders in: $ERROR_FOLDER"
            find "$ERROR_FOLDER" -type d -maxdepth 1 -regex ".*/SXM[1-8]_SN_[0-9]+" | while read -r sxm_folder; do
                analyze_sxm_subfolder "$sxm_folder" "$(echo "$id" | xargs)" &
            done
            # Wait for all parallel SXM sub-folder analyses to complete
            wait
        else
            echo "Skipping non-existent activity folder: '$ERROR_FOLDER' (Activity: '$activity')."
        fi
    done
fi
echo "-------------------------------------"
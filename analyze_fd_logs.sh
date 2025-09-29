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
# Call the new extraction function instead of the old tar command
if ! extract_archive "$LOG_ARCHIVE" "$TEMP_DIR"; then
    echo "Error: Failed to extract '$LOG_ARCHIVE'."
    exit 1
fi

# Construct the full path to run.log inside the extracted directory
# This finds the run.log and assumes its parent directory is the top-level extracted folder
RUN_LOG_PATH=$(find "$TEMP_DIR" -name "$LOG_FILE_IN_ARCHIVE" | head -n 1)

if [ -z "$RUN_LOG_PATH" ]; then
    echo "Error: '$LOG_FILE_IN_ARCHIVE' not found inside the extracted archive."
    exit 1
fi

echo "Found log file: $RUN_LOG_PATH"

# --- New early check for Final Result: PASS ---
if grep -q "Final Result: PASS" "$RUN_LOG_PATH"; then
    echo ""
    echo "--- Analysis Results for $LOG_FILE_IN_ARCHIVE ---"
    echo -e "${GREEN}Final Result: PASS found in log file. No further analysis required.${NC}"
    echo "-------------------------------------"
    exit 0 # Exit successfully
fi

# Determine the top-level directory where the archive was extracted (e.g., logs-YYYYMMDD-HHMMSS)
TOP_LEVEL_EXTRACTED_DIR=$(dirname "$RUN_LOG_PATH")

# Get all test lines that are NOT all zeros (i.e., "true issues")
# This filters out "MODS-000000000000" and "DGX-000000000000" lines
ALL_TEST_RESULTS=$(grep -E "^(MODS|DGX)-[0-9A-F]{12}" "$RUN_LOG_PATH")
TRUE_ISSUES=$(echo "$ALL_TEST_RESULTS" | grep -vE "^(MODS|DGX)-0{12}")

# Count PASS lines among true issues
PASS_COUNT=$(echo "$TRUE_ISSUES" | grep -E "\| OK$" | wc -l)

# Count total true issues
TOTAL_TRUE_ISSUES=$(echo "$TRUE_ISSUES" | wc -l)

# Calculate FAIL count among true issues
FAIL_COUNT=$((TOTAL_TRUE_ISSUES - PASS_COUNT))

# --- Report Results ---
echo ""
echo "--- Analysis Results for $LOG_FILE_IN_ARCHIVE ---"
echo "Total Relevant Test Entries (True Issues): $TOTAL_TRUE_ISSUES"
echo "PASS Count: $PASS_COUNT"
echo "FAIL Count: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo -e "--- ${RED}Failed Test Entries (True Issues)${NC} ---" # Print header in red
    # Print lines that are true issues but do NOT end with "| OK" in red
    FAILED_TRUE_ISSUES_LINES=$(echo "$TRUE_ISSUES" | grep -vE "\| OK$")
    echo -e "${RED}${FAILED_TRUE_ISSUES_LINES}${NC}" # Print lines in red

    echo ""
    echo "--- Detailed Error Codes from Failed Activities ---"
    echo "$FAILED_TRUE_ISSUES_LINES" | while IFS='|' read -r id activity_col rest_of_line; do
        # Extract the second column (activity) and trim whitespace
        activity=$(echo "$activity_col" | xargs)

        # Skip if activity starts with "Ext"
        if [[ "$activity" == "Ext"* ]]; then
            echo "Skipping activity '$activity' (excluded by filter)."
            continue
        fi

        ERROR_FOLDER="$TOP_LEVEL_EXTRACTED_DIR/$activity"

        if [ -d "$ERROR_FOLDER" ]; then
            echo "Searching in folder: $ERROR_FOLDER (for ID: $(echo "$id" | xargs))"
            # Search for "Error Code = " in all files within the folder
            # Then, filter out lines where the 12-digit code is all zeros
            ERROR_CODE_LINES=$(find "$ERROR_FOLDER" -type f -print0 | xargs -0 grep -E "Error Code = [0-9A-F]{12}" 2>/dev/null | grep -vE "Error Code = 0{12}")

            if [ -n "$ERROR_CODE_LINES" ]; then
                echo -e "${RED}$ERROR_CODE_LINES${NC}" # Print true error codes in red
            else
                echo "  No 'Error Code = ' (non-zero) found in files within '$ERROR_FOLDER'."
            fi
        else
            echo "Skipping non-existent activity folder: '$ERROR_FOLDER' (Activity: '$activity')."
        fi
    done
fi
echo "-------------------------------------"


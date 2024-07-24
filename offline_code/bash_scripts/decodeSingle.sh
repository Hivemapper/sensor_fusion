#!/bin/bash

# Input and output directories
inputDir="/Users/rogerberman/Desktop/4.9.9_DB_data/glistening-magenta-sculptor"
outputDir="${inputDir}/decoded"
recoveredDir="${inputDir}/recovered"

# Ensure output directories exist
mkdir -p "$outputDir"
mkdir -p "$recoveredDir"

# Function to decode and repair files
decode_and_repair_files() {
    local inputDir=$1
    local outputDir=$2

    # Loop through each file in the input directory
    for item in "$inputDir"/*; do
        if [ -f "$item" ]; then
            local itemName=$(basename "$item")
            local outputFile="$outputDir/$itemName"
            
            # Decode the file and save to output directory
            base64 --decode -i "$item" > "$outputFile"
            echo "Decoded: $outputFile"
            
            # Repair the decoded file as a SQLite database
            local recoveredFile="$recoveredDir/$itemName"
            
            # Repair the SQLite database
            sqlite3 "$outputFile" .recover > "${outputFile%.db}.sql"
            sqlite3 "$recoveredFile" < "${outputFile%.db}.sql"
            rm "${outputFile%.db}.sql"
            
            echo "Recovered: $recoveredFile"
        fi
    done
}

# Call function to decode and repair files
decode_and_repair_files "$inputDir" "$outputDir"

echo "Decoding and recovery completed."

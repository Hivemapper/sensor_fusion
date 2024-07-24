#!/bin/bash

#### Example usage: ./decodeDbs.sh /path/to/topLevelDir

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <top_level_directory>"
    exit 1
fi

# Top level input directory
topLevelDir="$1"

# Function to decode and repair files
decode_and_repair_files() {
    local inputDir=$1
    local outputDir=$2
    local recoveredDir=$3
    local directRepairDir=$4
    local finalOutputDir=$5

    # Ensure output directories exist
    mkdir -p "$outputDir"
    mkdir -p "$recoveredDir"
    mkdir -p "$directRepairDir"
    mkdir -p "$finalOutputDir"

    # Loop through each file in the input directory
    for item in "$inputDir"/*; do
        if [ -f "$item" ]; then
            local itemName=$(basename "$item")
            local outputFile="$outputDir/$itemName"
            local recoveredFile="$recoveredDir/$itemName"
            local directRepairFile="$directRepairDir/$itemName"
            
            # Decode the file and save to output directory
            base64 --decode -i "$item" > "$outputFile"
            echo "Decoded: $outputFile"
            
            # Repair the decoded file as a SQLite database
            sqlite3 "$outputFile" .recover > "${outputFile%.db}.sql"
            sqlite3 "$recoveredFile" < "${outputFile%.db}.sql"
            rm "${outputFile%.db}.sql"
            echo "Recovered: $recoveredFile"
            
            # Directly repair the original file
            sqlite3 "$item" .recover > "${directRepairFile%.db}.sql"
            sqlite3 "$directRepairFile" < "${directRepairFile%.db}.sql"
            rm "${directRepairFile%.db}.sql"
            echo "Directly Repaired: $directRepairFile"

            # Determine the largest file and copy it to the final output directory
            local largestFile="$outputFile"
            local largestSize=$(stat -f%z "$outputFile")

            if [ -f "$recoveredFile" ]; then
                local recoveredSize=$(stat -f%z "$recoveredFile")
                if [ $recoveredSize -gt $largestSize ]; then
                    largestFile="$recoveredFile"
                    largestSize=$recoveredSize
                fi
            fi

            if [ -f "$directRepairFile" ]; then
                local directRepairSize=$(stat -f%z "$directRepairFile")
                if [ $directRepairSize -gt $largestSize ]; then
                    largestFile="$directRepairFile"
                    largestSize=$directRepairSize
                fi
            fi

            cp "$largestFile" "$finalOutputDir/$itemName"
            echo "Copied $largestFile to $finalOutputDir/$itemName"
        fi
    done
}

# Loop through each subdirectory in the top level directory and process files
for subDir in "$topLevelDir"/*; do
    if [ -d "$subDir" ]; then
        subDirName=$(basename "$subDir")
        outputDir="${subDir}/decoded"
        recoveredDir="${subDir}/recovered"
        directRepairDir="${subDir}/direct_repair"
        finalOutputDir="${subDir}/output"

        # Call function to decode and repair files for each subdirectory
        decode_and_repair_files "$subDir" "$outputDir" "$recoveredDir" "$directRepairDir" "$finalOutputDir"
    fi
done

echo "Decoding, recovery, and final output generation completed for all subdirectories."



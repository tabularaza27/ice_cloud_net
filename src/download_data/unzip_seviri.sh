#!/bin/bash

# Check if user has provided an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory-path>"
    exit 1
fi

# Assign the input argument (directory path) to a variable
DIR_PATH=$1

# Specify an array of search strings (i.e. a file mask; with the current setting all files are matched)
search_strings=("NA-20")
# "NA-2010" "NA-2011" "NA-2012")

# Change to the specified directory
echo "Directory exists: $DIR_PATH"
cd "$DIR_PATH" || exit

# Loop through each search string
for search_string in "${search_strings[@]}"; do
    # Loop through zip files containing the current search string
    for zip_file in *"$search_string"*.zip; do
        if [ -e "$zip_file" ]; then
            # Extract the contents of the zip file
            unzip -o "$zip_file"

            # Remove the original zip file
            rm -f "$zip_file"

            echo "Unzipped and deleted: $zip_file"
        else
            echo "No zip files found with '$search_string' in the name."
        fi
    done
done
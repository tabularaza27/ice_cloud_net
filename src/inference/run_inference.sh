#!/bin/bash

# bash run_inference.sh 20100601 20100631

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 start_date end_date [python_script_path]"
    echo "Dates must be in the format YYYYMMDD"
    exit 1
fi

# Read the start and end dates from command line arguments
start_date=$1
end_date=$2

# If a third argument is provided, use it as the path to the python script, otherwise use the current directory
if [ -n "$3" ]; then
    python_script_path="$3/inference.py"
else
    python_script_path="./inference.py"
fi

# Ensure the python script exists
if [ ! -f "$python_script_path" ]; then
    echo "Error: Python script '$python_script_path' not found."
    exit 1
fi

# Convert dates to seconds since epoch for comparison
current_date=$(date -d "$start_date" +"%Y%m%d")
end_date=$(date -d "$end_date" +"%Y%m%d")

# Function to call the python script
call_inference() {
    local date_string=$1
    local timestep_slice=$2
    echo "Calling inference.py with date: $date_string and timestep_slice: $timestep_slice"
    python3 "$python_script_path" "$date_string" "$timestep_slice"
}

# Loop over the date range
while [ "$current_date" -le "$end_date" ]; do
    # Call inference.py twice for each day with different timestep slices
    call_inference "$current_date" "(0,48)"
    call_inference "$current_date" "(48,96)"

    # Move to the next day
    current_date=$(date -d "$current_date + 1 day" +"%Y%m%d")
done

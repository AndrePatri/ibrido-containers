#!/bin/bash

# Ensure the script is called with two arguments (basename and N)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <basename> <number_of_copies>"
    exit 1
fi

# Base name and number of copies to create
BASENAME=$1
N=$2

# Base directory name
BASE_DIR="${BASENAME}0"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' does not exist."
    exit 1
fi

# Loop to create the copies
for ((i=1; i<N; i++)); do
    NEW_DIR="${BASENAME}$i"
    if [ -d "$NEW_DIR" ]; then
        echo "Warning: Directory '$NEW_DIR' already exists. Skipping."
    else
        cp -r "$BASE_DIR" "$NEW_DIR"
        echo "Created: $NEW_DIR"
    fi
done
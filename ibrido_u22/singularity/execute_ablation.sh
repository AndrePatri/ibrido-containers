#!/bin/bash
# set -e # Uncomment to exit if any command fails

# Check if IBRIDO_CONTAINERS_PREFIX is set
if [ -z "$IBRIDO_CONTAINERS_PREFIX" ]; then
    echo "IBRIDO_CONTAINERS_PREFIX variable has not been set. Please set it to \${path_to_ibrido-containers}/ibrido_22/singularity."
    exit 1
fi

# Check if --cfg_dir is provided and is a valid directory
if [ -z "$1" ]; then
    echo "Please provide the --cfg_dir argument."
    exit 1
fi

cfg_dir="$2"

if [ ! -d "$cfg_dir" ]; then
    echo "The directory $cfg_dir is not valid. Please provide a valid directory."
    exit 1
fi

# Find the directory where the current script is located
script_dir=$(dirname "$0")

# Find all .sh files starting with "training_cfg_" in the specified directory
cfg_files=($(find "$cfg_dir" -maxdepth 1 -type f -name "training_cfg_*.sh"))

# Check if any files were found
if [ ${#cfg_files[@]} -eq 0 ]; then
    echo "No training configuration files found in $cfg_dir."
    exit 1
fi

# Print the list of found files (only the filenames, not the full path)
echo "Will run an ablation study with the configuration files:"
for file in "${cfg_files[@]}"; do
    # Use basename to strip the path and only print the file name
    echo "$(basename "$file")"
done

for file in "${cfg_files[@]}"; do
    # Use basename to strip the path and only print the file name
    file_name=$(basename "$file")
    # Run the ./execute command with the configuration file name
    "$script_dir/execute.sh" --cfg "$file_name"
done
echo "Ablation study completed."

#!/bin/bash

# Set base result directory path
base_result_dir="$(pwd)/ecm_result_dir"
echo "Base result directory: $base_result_dir"

# Check if base result directory exists, if not create it
if [ ! -d "$base_result_dir" ]; then
    mkdir -p "$base_result_dir"
    echo "Created base result directory: $base_result_dir"
fi

# Set mask directory path
mask="/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/combined_functional_localizer_masks/combined_mask_HFC.nii.gz"
echo "Mask: $mask"

# Create a unique result directory for the mask combination
result_dir="$base_result_dir"
echo "Result directory: $result_dir"

for dir in */; do
    # Check if the directory contains HCF.nii.gz file
    if [ -f "${dir}HFC_reg_f_MNI_f_MNI_2.nii.gz" ]; then
        # Extract subject name from directory path
        subject_name=$(basename "$dir")
        echo "Subject directory: $dir"
        echo "Subject name: $subject_name"

        # Execute vbcm command using preprocessed data
        vecm -in "${dir}HFC_reg_f_MNI_f_MNI_2.nii.gz" -out "$result_dir/vbcm_${subject_name}.v" -mask "$mask" -metric rlc
        vnifti -in "$result_dir/vbcm_${subject_name}.v" -out "$result_dir/vbcm_${subject_name}.nii"
        rm "$result_dir/vbcm_${subject_name}.v"
        echo "vbcm output saved to: $result_dir/vbcm_${subject_name}.v"
    fi
done

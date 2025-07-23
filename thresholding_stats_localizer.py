import os
import warnings
import sys
import csv
import numpy as np
import pandas as pd
from scipy import stats, signal # Combined imports for scipy
import matplotlib.pyplot as plt
import subprocess
from os.path import join as pjoin
from multiprocessing import Pool
import random
import nibabel as nib
from nilearn import masking, datasets, image
import seaborn as sns
import math

# --- Define Paths and Load Data ---
# Base directory for the project data
base_dir = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'

# Behavioral data paths
control_behav_path = os.path.join(base_dir, 'control_behavioral.csv')
als_behav_path = os.path.join(base_dir, 'ALS_behavioral.csv')

# Functional data directory
con_data_dir = os.path.join(base_dir, 'CON_AA_MNI_152T1_1mm_outputs_b3_masked11')
als_data_dir = os.path.join(base_dir, 'ALS_AA_MNI_152T1_1mm_outputs_b3_masked11') 

# Functional localizer mask directory
con_mask_dir = os.path.join(base_dir, 'CON_functional_localizer_masks')
als_mask_dir = os.path.join(base_dir, 'ALS_functional_localizer_masks')

# Load behavioral datasets for control and ALS
df_control_behav = pd.read_csv(control_behav_path)
df_als_behav = pd.read_csv(als_behav_path)

# Load subject IDs from text files
def load_subject_ids(file_path):
    """Loads a list of subject IDs from a tab-separated text file."""
    with open(file_path) as inf:
        reader = csv.reader(inf, delimiter="\t")
        return [row[0] for row in reader]

als_list_path = os.path.join(base_dir, 'als', 'als_list.txt')
control_list_path = os.path.join(base_dir, 'control', 'control_list.txt')

subj_als = load_subject_ids(als_list_path)
subj_control = load_subject_ids(control_list_path)

subjects = [subj_control, subj_als]
subject_groups = ['control', 'als']

# --- Data Loading Function ---
def datagrabber_als(group_ids, data_dir):
    """
    Grabs the file paths for fMRI BOLD data for a given list of subjects.

    Args:
        group_ids (list): A list of subject IDs.
        data_dir (str): The base directory containing subject data.

    Returns:
        list: A list of full file paths to the fMRI data files.
    """
    file_paths = []
    for sub_id in group_ids:
        # Assuming a consistent file naming convention
        filepath = pjoin(data_dir, sub_id, 'HFC_reg_f_MNI_f_MNI_2.nii.gz')
        if os.path.exists(filepath):
            file_paths.append(filepath)
        else:
            print(f"Warning: File not found for subject {sub_id} at {filepath}")
    return file_paths

# Get the list of fMRI data paths for the control group
run1_data_control = datagrabber_als(subj_control, con_data_dir)
# For ALS subjects
# run1_data_als = datagrabber_als(subj_als, als_data_dir)

# --- FSL-based Thresholding Functions ---
def fsl_out_to_float(output_string):
    """Extracts the first floating-point number from an FSL command's output string."""
    try:
        # Split the string and filter for numeric values
        threshold = [float(t) for t in output_string.split() if t.replace('.', '', 1).isdigit()]
        return threshold[0] if threshold else None
    except (ValueError, IndexError):
        return None

def threshold_and_binarize_mask(num_voxels, sub_id, body_part, data_dir, spmT_file):
    """
    Iteratively creates a binary mask for a specified number of top voxels
    using FSL commands.

    Args:
        num_voxels (int): The target number of voxels for the mask.
        sub_id (str): The subject ID.
        body_part (str): The body part being processed (e.g., 'l_hand').
        data_dir (str): The base directory for subject data.
        spmT_file (str): The name of the stat map file (e.g., 'spmT_0002.nii').
    """
    # Assuming the input file is a raw activation map
    subject_out_dir = pjoin(data_dir, sub_id)
    input_file = pjoin(subject_out_dir, f"sresampled_{body_part}_raw_coregistered.nii.gz")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found for subject {sub_id} - {body_part}. Skipping.")
        return

    # Get total number of non-zero voxels in the input file
    command_total_voxels = f'fslstats {input_file} -V'
    total_voxel_output = subprocess.check_output(command_total_voxels, shell=True, text=True)
    total_voxels = fsl_out_to_float(total_voxel_output)

    if total_voxels is None or total_voxels == 0:
        print(f"Warning: No valid voxels found in {input_file}. Skipping.")
        return

    # Calculate initial percentile for thresholding
    proportion = num_voxels / total_voxels
    percentile = 100 - 100 * proportion

    # Find the initial threshold value at the calculated percentile
    command_threshold_value = f'fslstats {input_file} -P {percentile}'
    thresh_output = subprocess.check_output(command_threshold_value, shell=True, text=True)
    current_threshold = fsl_out_to_float(thresh_output)

    if current_threshold is None:
        print(f"Error: Could not determine initial threshold for {input_file}. Skipping.")
        return

    binary_mask_path = pjoin(subject_out_dir, f'VOXEL_MASK_bin_{num_voxels}_{body_part}.nii.gz')

    # Iteratively adjust the threshold until the desired number of voxels is met
    while True:
        # Create a binary mask with the current threshold
        command_binarize = f'fslmaths {input_file} -thr {current_threshold} -bin {binary_mask_path}'
        subprocess.run(command_binarize, shell=True, check=True)

        # Count the number of voxels in the new binary mask
        command_count = f'fslstats {binary_mask_path} -V'
        voxel_count_output = subprocess.check_output(command_count, shell=True, text=True)
        voxel_count_in_mask = fsl_out_to_float(voxel_count_output)

        print(f"  - Current threshold: {current_threshold:.4f}, Voxels in mask: {int(voxel_count_in_mask)}")

        if voxel_count_in_mask >= num_voxels:
            print(f"  - Target of {num_voxels} voxels met for subject {sub_id}, {body_part}.")
            break  # Stop if the threshold meets the requirement
        
        # Reduce the threshold slightly if the voxel count is too low
        current_threshold *= 0.999
    
    print(f"Completed mask extraction for subject {sub_id}, {body_part}.")


the_voxels = [1501]
body_parts = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']

print("--- Starting iterative mask creation with FSL ---")
for num_voxels in the_voxels:
    for body_part in body_parts:
        print(f"\nProcessing {body_part} for {num_voxels} voxels...")
        for sub_id in subj_control:
            print(f"Processing subject {sub_id}...")
            
            threshold_and_binarize_mask(num_voxels, sub_id, body_part, con_data_dir, spmT_file='spmT_000x.nii')

# --- Timeseries Extraction with AFNI (3dmaskdump) ---


th_voxels = 1501

print("\n--- Starting timeseries extraction with AFNI (3dmaskdump) ---")

# First loop: For un-merged body parts
body_parts_unmerged = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']
for part in body_parts_unmerged:
    for sub_id, data_file in zip(subj_control, run1_data_control):
        print(f"Processing subject {sub_id} for part {part}...")
        
        subject_dir = pjoin(con_data_dir, sub_id)
        mask_path = pjoin(con_mask_dir, f"largest_cluster_mask_{part}_processed.nii.gz")
        
        resampled_mask_path = pjoin(subject_dir, f"sresampled_mask_{part}_{th_voxels}.nii.gz")
        timeseries_output_path = pjoin(subject_dir, f"sadata_timeseries_{part}_{th_voxels}.txt")
        
        if os.path.exists(timeseries_output_path):
            print(f"  - Skipping {sub_id} - {part}, timeseries already exists.")
            continue
        
        # Resample mask to match functional data resolution
        subprocess.run(['3dresample', '-master', data_file, '-input', mask_path, '-prefix', resampled_mask_path], check=True)
        
        # Extract timeseries
        subprocess.run(['3dmaskdump', '-mask', resampled_mask_path, '-o', timeseries_output_path, data_file], check=True)
        print("  - Timeseries extraction complete.")

# Second loop: For merged body parts
body_parts_merged = ['hand', 'foot', 'tongue']
for part in body_parts_merged:
    for sub_id, data_file in zip(subj_control, run1_data_control):
        print(f"Processing subject {sub_id} for merged part {part}...")

        subject_dir = pjoin(con_data_dir, sub_id)
        mask_path = pjoin(con_mask_dir, f"largest_cluster_mask_{part}_merged.nii.gz")

        resampled_mask_path = pjoin(subject_dir, f"sresampled_mask_{part}_{th_voxels}_merged.nii.gz")
        timeseries_output_path = pjoin(subject_dir, f"sadata_timeseries_{part}_{th_voxels}_merged.dat")

        if os.path.exists(timeseries_output_path):
            print(f"  - Skipping {sub_id} - {part}, timeseries already exists.")
            continue

        # Resample mask
        subprocess.run(['3dresample', '-master', data_file, '-input', mask_path, '-prefix', resampled_mask_path], check=True)
        
        # Extract timeseries
        subprocess.run(['3dmaskdump', '-mask', resampled_mask_path, '-o', timeseries_output_path, data_file], check=True)
        print("  - Timeseries extraction complete.")

# Third loop: For whole-brain (WB) functional mask
print("\nProcessing whole-brain functional mask...")
for sub_id, data_file in zip(subj_control, run1_data_control):
    print(f"Processing subject {sub_id} for whole-brain mask...")
    
    subject_dir = pjoin(con_data_dir, sub_id)
    mask_path = pjoin(con_mask_dir, "wb_functional_HFC.nii.gz")
    
    resampled_mask_path = pjoin(subject_dir, "sresampled_mask_WB_functional_1501_merged.nii.gz")
    timeseries_output_path = pjoin(subject_dir, "sadata_timeseries_WB_functional_1501_merged.dat")
    
    if os.path.exists(timeseries_output_path):
        print(f"  - Skipping {sub_id}, timeseries already exists.")
        continue

    # Resample mask
    subprocess.run(['3dresample', '-master', data_file, '-input', mask_path, '-prefix', resampled_mask_path], check=True)
    
    # Extract timeseries
    subprocess.run(['3dmaskdump', '-mask', resampled_mask_path, '-o', timeseries_output_path, data_file], check=True)
    print("  - Timeseries extraction complete.")

print("\nAll timeseries extraction tasks finished.")
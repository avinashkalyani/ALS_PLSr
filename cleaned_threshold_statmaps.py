"""
Script for thresholding fMRI statistical maps using FSL,
and extracting time series from thresholded binary masks.
"""

import os
import csv
import subprocess
from os.path import join as pjoin
import pandas as pd

# ========== Configuration ==========
base_path = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'
ds_dir = pjoin(base_path, 'CON_AA_MNI_152T1_1mm_outputs_b3_masked11')
mask_dir = pjoin(base_path, 'CON_functional_localizer_masks')
th_voxels = 1501
body_parts = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']
spmT_files = ['spmT_0002.nii', 'spmT_0003.nii', 'spmT_0004.nii', 'spmT_0005.nii', 'spmT_0006.nii']

# ========== Load Subject Lists ==========
def load_subjects(txt_file):
    with open(txt_file) as f:
        return list(zip(*csv.reader(f, delimiter="\t")))[0]

control_subjects = load_subjects(pjoin(base_path, 'control/control_list.txt'))

# ========== Threshold Utilities ==========

def fsl_output_to_threshold(output):
    values = [float(t) for t in ''.join(output).split() if t.replace('.', '').isdigit()]
    return values[0] if values else None

def threshold_stat_map(th_voxels, sub_ids, stat_file, body_part):
    for sub_id in sub_ids:
        subj_dir = pjoin(ds_dir, sub_id)
        stat_path = pjoin(subj_dir, f"sresampled_{body_part}_raw_coregistered.nii.gz")
        mask_output = pjoin(subj_dir, f"VOXEL_MASK_bin_{th_voxels}_{body_part}.nii.gz")

        # Initial voxel count
        vox_output = subprocess.check_output(f"fslstats {stat_path} -V", shell=True, text=True)
        total_vox = fsl_output_to_threshold(vox_output)

        # Calculate percentile threshold
        prop = th_voxels / total_vox
        perc = 100 - 100 * prop
        perc_output = subprocess.check_output(f"fslstats {stat_path} -a -P {perc}", shell=True, text=True)
        TH = fsl_output_to_threshold(perc_output)

        # Apply threshold
        while True:
            subprocess.run(f"fslmaths {stat_path} -thr {TH} -bin {mask_output}", shell=True, check=True)
            count_output = subprocess.check_output(f"fslstats {mask_output} -V", shell=True, text=True)
            current_vox = int(count_output.split()[0])
            if current_vox >= th_voxels:
                break
            TH *= 0.999  # Slightly reduce threshold if not enough voxels

        print(f"Done thresholding {body_part} for {sub_id} with final voxels: {current_vox}")

# ========== Run Thresholding ==========
for idx, stat_file in enumerate(spmT_files):
    threshold_stat_map(th_voxels, control_subjects, stat_file, body_parts[idx])

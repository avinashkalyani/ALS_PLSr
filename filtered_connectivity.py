import nibabel as nib
import numpy as np
import pylab as p
from nilearn import input_data, plotting
import os
import csv
from scipy import stats
import time

from nilearn import plotting, signal
"""importing als """

with open('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/als/als_list.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    als = list(zip(*reader))[0]

"""importing control """
with open('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/control/control_list.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    control = list(zip(*reader))[0]

base_dir = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'
group_dirs = ['CON_AA_MNI_152T1_1mm_outputs_b3_masked11','ALS_AA_MNI_152T1_1mm_outputs_b3_masked11']
gm_path = os.path.join(base_dir, 'CrebrA_masks/wb_mask.nii')
seeds = ['l_hand_seed.nii','r_hand_seed.nii.gz','l_foot_seed.nii.gz','r_foot_seed.nii.gz', 'tongue_seed.nii.gz']

# get onsets then filter time series


#load the timeseries masked already

def generate_correlation_matrices(fMRI_path, gm_path, seed_coords, seed_radius, out_con_matrix, labels_onset, body_idx, gm_time_series,gm_masker):
    """
    Perform seed-based correlation analysis and save correlation matrices.

    Parameters:
    - fMRI_path (str): Path to the preprocessed fMRI data.
    - gm_path (str): Path to the gray matter mask.
    - seed_coords (tuple): Coordinates of the seed.
    - seed_radius (int): Radius of the seed.
    - out_con_matrix (str): Path to save the output correlation matrix.
    """
    # Load the preprocessed fMRI data
    fmri_img = nib.load(fMRI_path)

    # Load the GM mask
    gm_mask_img = nib.load(gm_path)

    # Define seed mask using coordinates and radius
    seed_masker = input_data.NiftiSpheresMasker(
        seeds=[seed_coords],
        radius=seed_radius,
        mask_img=gm_mask_img,
        standardize=True)

    # Extract time series from the seed mask
    seed_time_series = seed_masker.fit_transform(fmri_img)

    # Band-pass filtering

    seed_time_series_filtered = signal.clean(seed_time_series, detrend=False, standardize=True, t_r=2,
                                             low_pass=0.1, high_pass=0.01, confounds=None)
    seed_time_series_filt = np.nan_to_num(stats.zscore(filter_timeseries(seed_time_series_filtered, labels_onset, body_idx),axis=0))
    # Average the filtered seed time series
    #seed_time_series_avg = np.mean(seed_time_series_filtered, axis=1, keepdims=True)

    # Extract time series from the GM mask
    #gm_masker = input_data.NiftiMasker(mask_img=gm_mask_img, standardize=True)

    #gm_time_series = gm_masker.fit_transform(fmri_img)
    print('Filtering now')
    # Band-pass filtering

    gm_time_series_filtered = signal.clean(gm_time_series, detrend=False, standardize=True, t_r=2,
                                           low_pass=0.1, high_pass=0.01, confounds=None)

    gm_time_series_filt = np.nan_to_num(stats.zscore(filter_timeseries(gm_time_series_filtered, labels_onset, body_idx),axis=0))

    print('correlation now')
    # Compute Pearson correlation between seed region and each voxel in the GM mask
    #seed_gm_correlation = np.corrcoef( gm_time_series_filtered,seed_time_series_filtered)
    #stim_seed_time_series_filtered = filter_timeseries(seed_time_series_filtered,labels_onset,body_idx)
    #stim_gm_time_series_filtered = filter_timeseries(gm_time_series_filtered, labels_onset, body_idx)
    print( 'shapes',seed_time_series_filt.shape[0],  'shape_gm',gm_time_series_filt.shape[0])
    seed_gm_correlation = np.dot(seed_time_series_filt.T, gm_time_series_filt) / seed_time_series_filt.shape[0]
    print('saving the correlation matrix', seed_gm_correlation.shape)
    # Reshape seed_gm_correlation to match GM mask shape
    seed_gm_correlation_img = gm_masker.inverse_transform(seed_gm_correlation)

    # Save the seed-based connectivity map
    nib.save(seed_gm_correlation_img, out_con_matrix)
    print('Matrix saved at', out_con_matrix)
    return seed_gm_correlation


def get_voxel_MNI_coordinates(seed_mask_path):
    """
    Get the MNI coordinates of the voxel in the seed mask.

    Parameters:
    - seed_mask_path (str): Path to the seed mask Nifti file.

    Returns:
    - Tuple of MNI coordinates (x, y, z).
    """
    # Load the Nifti image
    seed_img = nib.load(seed_mask_path)
    data = seed_img.get_fdata()

    # Find the coordinates of the active voxel
    voxel_coords = np.argwhere(data != 0)[0]

    # Get the affine transformation matrix
    affine = seed_img.affine

    # Convert voxel coordinates to MNI coordinates
    mni_coords = nib.affines.apply_affine(affine, voxel_coords)

    return tuple(mni_coords)

# Define seed radius
seed_radius = 8

body_part = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']

name_ons_body = ['HandLeft', 'HandRight','FootLeft', 'FootRight', 'Tongue'] # it was differnt before

def get_onsets_randomruns(subject_group,
                          ds_dir, group):
    # array with onsets of shape ngroup nparts, nsub
    onsets_array = np.zeros(shape=(12, 5, 4))
    for sub_idx, sub_id in enumerate(subject_group):
        for body_idx, body_int in enumerate(name_ons_body):
            dig_abspath = os.path.join(ds_dir, group, sub_id, 'functional/time%s.ons' % body_int)
            with open(dig_abspath, 'r') as f:
                csv_reader = csv.reader(f, delimiter='\n')
                dig_onsets = [float(row[0]) for row in csv_reader]
                #print(sub_id, body_idx)
                onsets_array[sub_idx, body_idx] = dig_onsets
    return onsets_array

ds_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project"

# onsets to labels:

def randomruns_onsets_to_labels(randomruns_onsets_array,
                                stimdur=12,
                                tr=2.0,
                                nvols_in_random_run=302):
    stimdur_ms = int(stimdur * 100)
    nsubs, ndigits, nonsets_per_digit = randomruns_onsets_array.shape
    labels_ms = np.zeros(shape=(nsubs, int(nvols_in_random_run * tr * 100)))
    for sub_idx in range(nsubs):
        for digit_idx, digit_id in enumerate(range(1, ndigits + 1)):
            dig_onsets = randomruns_onsets_array[sub_idx, digit_idx]
            for ons in dig_onsets:
                ons_ms = int(ons * 100)
                labels_ms[sub_idx, int(ons_ms):int(ons_ms + stimdur_ms)] = digit_id
    random_labels = labels_ms[:, ::int(100 * tr)]
    return random_labels

#timeseries : vox by time dimension
def filter_timeseries(timeseries, labels_onset, body_idx):
    filtered_indxs = np.where(p.isin(labels_onset, body_idx))[0]
    filtered_timeseries = timeseries[filtered_indxs]
    return filtered_timeseries

groups = ["control", "ALS"]
seed_con_mat = []
import time


#filter the timepoints


#do seed based connectivity
#Fischer Z transform

#statistical tests to see the group significance and seed level significant
# fid the p values voxel specific: like before


# Start the loop

for gr_idx, sub_group in enumerate([control, als]):
    working_dir = group_dirs[gr_idx]
    seed_con_mat_group = []
    start_time = time.time()
    for sub_idx, sub in enumerate(sub_group):
        fMRI_path = os.path.join(base_dir, working_dir, f'{sub}/HFC_reg_f_MNI_f_MNI_2.nii.gz')
        out_con_matrix_sub = os.path.join(base_dir, working_dir, f'{sub}/')
        onsets_array = get_onsets_randomruns(sub_group, ds_dir, groups[gr_idx])
        labels_onset = randomruns_onsets_to_labels(onsets_array)
        gm_mask_img = nib.load(gm_path)
        gm_masker = input_data.NiftiMasker(mask_img=gm_mask_img, smoothing_fwhm = 3 , standardize=True)
        gm_time_series = gm_masker.fit_transform(nib.load(fMRI_path))
        seed_con_mat_sub = []
        for seed_idx, seed in enumerate(seeds):
            seed_mask_path = os.path.join(base_dir, f'CrebrA_masks/{seed}')
            seed_coords = get_voxel_MNI_coordinates(seed_mask_path)
            out_con_matrix = os.path.join(out_con_matrix_sub, f'smooth_WB_body_filtered_connectivity_{seed}.nii.gz')
            body_idx = seed_idx + 1

            print("body_idx", body_idx, "seed", seed)

            start_time = time.time()  # Start timing for each iteration
            con_matrix = generate_correlation_matrices(fMRI_path, gm_path, seed_coords, seed_radius,
                                                       out_con_matrix=out_con_matrix,
                                                       labels_onset=labels_onset[sub_idx], body_idx=body_idx, gm_time_series = gm_time_series, gm_masker=gm_masker)
            end_time = time.time()
            time_taken = end_time - start_time
            print('dim:', con_matrix.shape, f'time:{time_taken:.6f} seconds')


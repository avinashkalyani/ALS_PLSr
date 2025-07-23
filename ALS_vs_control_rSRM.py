import os
import warnings
import sys
import time
import csv
from os.path import join as pjoin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn import svm
from sklearn.model_selection import permutation_test_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from brainiak.funcalign.rsrm import RSRM

# --- Configuration and Data Loading ---
# Define base directory for data
BASE_DIR = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project'
RESULTS_DIR = pjoin(BASE_DIR, 'results')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load subject lists
def load_subject_list(filepath):
    """Loads a list of subjects from a tab-separated file."""
    with open(filepath) as f:
        reader = csv.reader(f, delimiter="\t")
        return list(zip(*reader))[0]

control_subjects = load_subject_list(pjoin(BASE_DIR, 'control', 'control_list.txt'))
als_subjects = load_subject_list(pjoin(BASE_DIR, 'als', 'als_list.txt'))

all_subjects = [control_subjects, als_subjects]
subject_group_names = ['control', 'als']

# Create numerical labels for classification
labels_control = np.zeros(len(control_subjects))
labels_als = np.ones(len(als_subjects))
all_labels = np.concatenate((labels_control, labels_als)).tolist()

# Define body parts for data organization
BODY_PARTS = ['HandLeft', 'HandRight', 'FootLeft', 'FootRight', 'Tongue']

# File paths for preprocessed data
NPZ_FILES = {
    'rhand': pjoin(RESULTS_DIR, "all_arrs_rhand.npz"),
    'lhand': pjoin(RESULTS_DIR, "all_arrs_lhand.npz"),
    'rfoot': pjoin(RESULTS_DIR, "all_arrs_rfoot.npz"),
    'lfoot': pjoin(RESULTS_DIR, "all_arrs_lfoot.npz"),
    'face': pjoin(RESULTS_DIR, "all_arrs_face.npz"),
}

def load_all_body_part_data(npz_files):
    """Loads and returns all preprocessed data arrays from NPZ files."""
    loaded_data = {}
    for key, filepath in npz_files.items():
        with np.load(filepath, allow_pickle=True) as data:
            loaded_data[key] = [data[k] for k in data]
    return loaded_data

all_arrs_raw = load_all_body_part_data(NPZ_FILES)

# Reorder loaded data to match BODY_PARTS for easier indexing
all_arrs_ordered = [
    all_arrs_raw['lhand'],  # Corresponds to HandLeft (index 0)
    all_arrs_raw['rhand'],  # Corresponds to HandRight (index 1)
    all_arrs_raw['lfoot'],  # Corresponds to FootLeft (index 2)
    all_arrs_raw['rfoot'],  # Corresponds to FootRight (index 3)
    all_arrs_raw['face']    # Corresponds to Tongue/Face (index 4)
]

# --- Data Preprocessing and Arrangement ---
def preprocess_array(arr, target_shape=(1500, 302)):
    """
    Preprocesses a single array to have a specified target shape by zero-padding or cropping.
    """
    if arr.shape == target_shape:
        return arr
    pad_width = ((0, max(0, target_shape[0] - arr.shape[0])),
                 (0, max(0, target_shape[1] - arr.shape[1])))
    padded_arr = np.pad(arr, pad_width, mode='constant')
    return padded_arr[:target_shape[0], :target_shape[1]]

def arrange_data_by_onset_group(all_data_arrays, onset_indices, target_shape=(1500, 302), is_als=False, apply_zscore=False):
    """
    Arranges data into 'first affected' or 'later affected' groups based on onset indices.
    Applies preprocessing and optional z-scoring.
    """
    arranged_data = []
    for sub_idx, onset_info in enumerate(onset_indices):
        subject_data_parts = []
        # Determine which group's data to pick (control=0, als=1)
        group_data_index = 1 if is_als else 0

        if isinstance(onset_info, list):
            for part_idx in onset_info:
                data_to_process = all_data_arrays[part_idx][group_data_index][sub_idx][:1500, :]
                processed = preprocess_array(data_to_process, target_shape)
                if apply_zscore:
                    # zscore is applied column-wise (across TRs for each voxel)
                    processed = np.nan_to_num(stats.zscore(processed, axis=1))
                subject_data_parts.append(processed)
            arranged_data.append(np.concatenate(subject_data_parts, axis=0))
        elif isinstance(onset_info, int):
            data_to_process = all_data_arrays[onset_info][group_data_index][sub_idx][:1500, :]
            processed = preprocess_array(data_to_process, target_shape)
            if apply_zscore:
                processed = np.nan_to_num(stats.zscore(processed, axis=1))
            arranged_data.append(processed)
    return arranged_data

# Onset indices for 'first affected' group for ALS subjects
# This array indicates which body part was first affected for each of the 11 ALS subjects
# (0=LHand, 1=RHand, 2=LFoot, 3=RFoot, 4=Face/Tongue)
first_onset_indices_als = [0, 2, 2, 1, 4, 0, 1, 4, 1, 4, [0, 1]]

# Generate 'later affected' indices for ALS subjects
def generate_later_affected_indices(first_onset_list, all_body_part_indices=range(5)):
    """Generates indices for body parts NOT in the first affected list."""
    later_affected_indices = []
    for item in first_onset_list:
        if isinstance(item, list):
            later_affected_indices.append([i for i in all_body_part_indices if i not in item])
        else:
            later_affected_indices.append([i for i in all_body_part_indices if i != item])
    return later_affected_indices

later_affected_indices_als = generate_later_affected_indices(first_onset_indices_als)

# Arrange data for "first affected" and "later affected" groups
# For controls, 'first affected' and 'later affected' groups are simply based on the ALS subject's
# first/later affected body parts, to create comparable groups.

# Data for 'first affected' analysis (no z-scoring at this stage, happens later if cyclic)
arrs_als_first = arrange_data_by_onset_group(all_arrs_ordered, first_onset_indices_als, is_als=True, apply_zscore=False)
arrs_con_first = arrange_data_by_onset_group(all_arrs_ordered, first_onset_indices_als, is_als=False, apply_zscore=False)

# Data for 'later affected' analysis (no z-scoring at this stage for the full array, happens within arrange_segments_from_onsets if applied to cyclic data)
arrs_als_later = arrange_data_by_onset_group(all_arrs_ordered, later_affected_indices_als, is_als=True, apply_zscore=False)
arrs_con_later = arrange_data_by_onset_group(all_arrs_ordered, later_affected_indices_als, is_als=False, apply_zscore=False)


# --- Synchronization (Timing Files) and Data Splitting ---

def get_onsets_randomruns(subject_list, base_dir, group_name, body_parts=BODY_PARTS):
    """
    Reads onset timing files for each subject and body part.
    Returns a numpy array of shape (n_subjects, n_body_parts, n_onsets_per_part).
    """
    n_subs = len(subject_list)
    n_parts = len(body_parts)
    # Assuming a fixed number of onsets per digit/body part (4 as seen in original code)
    onsets_array = np.zeros(shape=(n_subs, n_parts, 4))
    for sub_idx, sub_id in enumerate(subject_list):
        for body_idx, body_part in enumerate(body_parts):
            onset_filepath = pjoin(base_dir, group_name, sub_id, f'functional/time{body_part}.ons')
            try:
                with open(onset_filepath, 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\n')
                    # Ensure onsets are floats and handle potential empty lines/malformed data
                    dig_onsets = [float(row[0]) for row in csv_reader if row and row[0].strip()]
                    # Pad with NaN if fewer than 4 onsets, or truncate if more
                    onsets_array[sub_idx, body_idx, :] = np.pad(dig_onsets, (0, 4 - len(dig_onsets)), 'constant', constant_values=np.nan)[:4]
            except FileNotFoundError:
                print(f"Warning: Onset file not found for {sub_id}, {body_part}. Skipping.")
                onsets_array[sub_idx, body_idx, :] = np.nan # Mark as NaN if file missing
            except ValueError:
                print(f"Warning: Could not parse onset data for {sub_id}, {body_part}. Skipping.")
                onsets_array[sub_idx, body_idx, :] = np.nan # Mark as NaN if parsing fails
    return onsets_array

def arrange_segments_from_onsets(subject_list, onsets_array, run_data_arrays, dur_trs=7, apply_zscore=False):
    """
    Arranges fMRI data segments based on onset timings, creating a 'cyclic' array.
    This function extracts time series data around each onset and optionally z-scores.
    """
    arranged_data_list = []
    n_digits_conditions = len(BODY_PARTS) # 5 body parts / conditions
    
    for sub_idx, sub_id in enumerate(subject_list):
        data_onsets_df = pd.DataFrame(onsets_array[sub_idx].T, columns=[f'D_{i+1}' for i in range(n_digits_conditions)])
        # Convert onset times (seconds) to TR indices (TR=2.0s)
        tr_indices_df = np.ceil(data_onsets_df / 2).astype(int)

        subject_run_arr = run_data_arrays[sub_idx]
        n_voxels, n_trs_total = subject_run_arr.shape # run_data_arrays are (voxels, TRs)

        # Initialize array to hold binned TRs: (n_conditions, n_onsets_per_condition, duration_trs, n_voxels)
        binned_trs = np.zeros((n_digits_conditions, 4, dur_trs, n_voxels))

        for digit_idx, digit_col_name in enumerate(tr_indices_df.columns):
            for onset_instance_idx in range(4): # 4 onsets per digit/condition
                tr_start_idx = tr_indices_df.iloc[onset_instance_idx][digit_col_name]
                
                # Ensure we don't go out of bounds and onset is not NaN
                if not np.isnan(tr_start_idx) and tr_start_idx >= 0 and (tr_start_idx + dur_trs) <= n_trs_total:
                    binned_trs[digit_idx, onset_instance_idx, :, :] = subject_run_arr.T[tr_start_idx : tr_start_idx + dur_trs, :]
                else:
                    # Fill with NaN if data is missing or out of bounds for this segment
                    binned_trs[digit_idx, onset_instance_idx, :, :] = np.nan

        # Reshape into a 'cyclic' array (concatenating all segments for the subject)
        cyclic_data = binned_trs.reshape(-1, n_voxels)
        # Filter out NaN rows that resulted from missing onset data
        cyclic_data = cyclic_data[~np.isnan(cyclic_data).all(axis=1)] 
        
        if apply_zscore:
            cyclic_data = np.nan_to_num(stats.zscore(cyclic_data, axis=0)) # Z-score along columns (TRs) for each voxel

        arranged_data_list.append(cyclic_data.T) # Transpose back to (voxels, time) for SRM

    return arranged_data_list

# Split data into two runs (e.g., first half and second half of time points)
# For the initial (non-cyclic) data, this is a direct split of the TRs.
def split_data_into_runs(data_list, split_point):
    """Splits each array in a list into two runs."""
    run1 = [d[:, :split_point] for d in data_list]
    run2 = [d[:, split_point:] for d in data_list]
    return run1, run2

# Define split points for the two different types of data (full vs. cyclic)
SPLIT_POINT_FULL_DATA = 151 # for (1500, 302) data, splits into (1500, 151) and (1500, 151)
SPLIT_POINT_CYCLIC_DATA = 70 # for (N_voxels, 140) data (5 conditions * 4 onsets * 7 TRs/onset = 140 TRs)

# --- rSRM Cross-Validation and Projection (Corrected based on original logic) ---
def run_crossval_srm_projection_corrected(
    control_run1_data, control_run2_data,  # Control group data (Run 1, Run 2)
    als_run1_data, als_run2_data,          # ALS group data (Run 1, Run 2)
    k_features, output_tag, n_iterations=30
):
    """
    Performs rSRM alignment and projection. For each test subject, their data is projected
    into *two* shared spaces: one trained on the *control* group's training data,
    and one trained on the *ALS* group's training data.
    The two resulting projected arrays are then concatenated.
    """
    n_subjects = len(control_run1_data) # Assuming same number of subjects in both groups
    
    # These lists will store the projected data for both runs, for both groups.
    # Structure: [group_idx][run_idx][subject_idx] -> projected_data
    # Where group_idx 0 = Control, 1 = ALS
    projected_data_by_group_and_run = [
        [[] for _ in range(n_subjects)], # For Control (group 0), Run 1
        [[] for _ in range(n_subjects)], # For Control (group 0), Run 2
        [[] for _ in range(n_subjects)], # For ALS (group 1), Run 1
        [[] for _ in range(n_subjects)]  # For ALS (group 1), Run 2
    ]

    # Iterate over the two 'training runs' (Run 1 and Run 2)
    for train_run_idx in range(2): # 0 for Run 1, 1 for Run 2
        # Select the training data for control and ALS for the current training run
        # train_run_idx = 0 -> use run1_data for training; train_run_idx = 1 -> use run2_data for training
        current_control_training_data = (control_run1_data, control_run2_data)[train_run_idx]
        current_als_training_data = (als_run1_data, als_run2_data)[train_run_idx]
        
        # Select the test data for both groups for the *other* run (cross-run prediction)
        # abs(train_run_idx - 1) means if train_run_idx is 0 (Run 1), test_run_idx is 1 (Run 2) and vice versa.
        current_control_test_data = (control_run1_data, control_run2_data)[abs(train_run_idx - 1)]
        current_als_test_data = (als_run1_data, als_run2_data)[abs(train_run_idx - 1)]

        for test_sub_idx in range(n_subjects):
            start = time.time()
            print(f'Training with Run {train_run_idx + 1}, processing test subject {test_sub_idx + 1}/{n_subjects}')

            # 1. Train SRM for Control Group (excluding the current test subject)
            # This SRM learns the shared space of the *control* group from their training data (run 1 or run 2)
            train_subs_control = [x for i, x in enumerate(current_control_training_data) if i != test_sub_idx]
            srm_control = RSRM(n_iter=n_iterations, features=k_features)
            srm_control.fit(train_subs_control)

            # 2. Train SRM for ALS Group (excluding the current test subject)
            # This SRM learns the shared space of the *ALS* group from their training data (run 1 or run 2)
            train_subs_als = [x for i, x in enumerate(current_als_training_data) if i != test_sub_idx]
            srm_als = RSRM(n_iter=n_iterations, features=k_features)
            srm_als.fit(train_subs_als)

            # 3. Project test subject's data for *Control Group*
            # Take the test subject's data from the *test run* (the other run)
            test_sub_control_data = current_control_test_data[test_sub_idx]
            
            # Project this test subject's data into the shared space of CONTROL group
            shared_control_proj_from_control_srm, _ = srm_control.transform_subject(test_sub_control_data)
            # Project this test subject's data into the shared space of ALS group
            shared_control_proj_from_als_srm, _ = srm_als.transform_subject(test_sub_control_data)

            # Concatenate the two projections for the control test subject
            projected_data_by_group_and_run[train_run_idx][test_sub_idx] = np.concatenate(
                (shared_control_proj_from_control_srm, shared_control_proj_from_als_srm), axis=1
            )
            
            # 4. Project test subject's data for *ALS Group*
            test_sub_als_data = current_als_test_data[test_sub_idx]
            
            # Project this test subject's data into the shared space of CONTROL group
            shared_als_proj_from_control_srm, _ = srm_control.transform_subject(test_sub_als_data)
            # Project this test subject's data into the shared space of ALS group
            shared_als_proj_from_als_srm, _ = srm_als.transform_subject(test_sub_als_data)

            # Concatenate the two projections for the ALS test subject
            projected_data_by_group_and_run[train_run_idx + 2][test_sub_idx] = np.concatenate(
                (shared_als_proj_from_control_srm, shared_als_proj_from_als_srm), axis=1
            )
            
            elapsed = time.time() - start
            print(f'This subject took: {elapsed:.2f} seconds')
            
    
    # Save for Control (age_idx = 0)
    np.savez(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_{output_tag}_0_{k_features}.npz'),
             projected_data=[projected_data_by_group_and_run[0], projected_data_by_group_and_run[1]],
             allow_pickle=True)
    
    # Save for ALS (age_idx = 1)
    np.savez(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_{output_tag}_1_{k_features}.npz'),
             projected_data=[projected_data_by_group_and_run[2], projected_data_by_group_and_run[3]],
             allow_pickle=True)

    print(f"SRM projection complete for {output_tag}.")

# --- Classification and Evaluation ---
def loo_classify_balanced_combination(g1_data, g2_data, all_subject_labels):
    """
    Performs Leave-One-Out (LOO) cross-validation classification using an SVM.
    Assumes balanced groups (same number of subjects in g1 and g2).
    Includes permutation testing for significance.
    """
    loo = LeaveOneOut()
    
    # Flatten data for classification
    X1_flat = [d.flatten() for d in g1_data]
    X2_flat = [d.flatten() for d in g2_data]

    all_scores = [] # Stores accuracy for each LOO fold
    
    # LOO for g1 subjects
    for train_idx1, test_idx1 in loo.split(X1_flat):
        X1_train, X1_test = np.array(X1_flat)[train_idx1], np.array(X1_flat)[test_idx1]
        y1_train, y1_test = np.array(labels_control)[train_idx1], np.array(labels_control)[test_idx1]
        
        # Nested LOO for g2 subjects to ensure balanced test sets (one from g1, one from g2)
        for train_idx2, test_idx2 in loo.split(X2_flat):
            X2_train, X2_test = np.array(X2_flat)[train_idx2], np.array(X2_flat)[test_idx2]
            y2_train, y2_test = np.array(labels_als)[train_idx2], np.array(labels_als)[test_idx2]

            # Combine training and test sets
            X_train = np.concatenate((X1_train, X2_train))
            y_train = np.concatenate((y1_train, y2_train))
            X_test = np.concatenate((X1_test, X2_test))
            y_test = np.concatenate((y1_test, y2_test))

            classifier = svm.SVC(kernel='linear')
            classifier.fit(X_train, y_train)
            predicted_labels = classifier.predict(X_test)
            score = accuracy_score(y_test, predicted_labels)
            all_scores.append(score)

    mean_accuracy = np.mean(all_scores)
    std_accuracy = np.std(all_scores)
    sem_accuracy = std_accuracy / np.sqrt(len(all_scores))
    
    print(f"Mean LOO accuracy: {mean_accuracy:.4f} � {sem_accuracy:.4f}")

    # Permutation Test
    # For permutation test, we use the full combined dataset (X1_flat + X2_flat)
    X_combined = np.concatenate((X1_flat, X2_flat), axis=0)
    y_combined = np.array(all_subject_labels) # Use the full set of labels (control=0, als=1)

    cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Using a fixed random state for reproducibility
    
    # Calculate permutation test score
    # Note: `permutation_test_score` runs its own internal cross-validation for the observed score
    # and then permutes labels. The `classifier` passed here is a fresh instance for consistency.
    cv_score_observed, perm_scores, p_value = permutation_test_score(
        svm.SVC(kernel='linear'), X_combined, y_combined,
        scoring="accuracy", cv=cv_stratified, n_permutations=1000
    )
    
    print(f"Observed CV score (permutation test): {cv_score_observed:.4f}")
    print(f"Permutation test p-value: {p_value:.4f}")

    return all_scores, mean_accuracy, std_accuracy, sem_accuracy, cv_score_observed, perm_scores, p_value

def plot_confusion_matrix_and_permutation_test(
    true_labels, predicted_labels, observed_score, permutation_scores, p_value, title_prefix=""
):
    """
    Plots a confusion matrix and a histogram of permutation test scores.
    """
    # Create the 'figures' directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "ALS"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    disp.plot(cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f"{title_prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(pjoin('figures', f'confusion_matrix_{title_prefix}.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Permutation Test Histogram
    fig_perm, ax_perm = plt.subplots(figsize=(7, 5))
    sns.histplot(permutation_scores, bins=20, kde=True, color='gray', ax=ax_perm)
    ax_perm.axvline(observed_score, color='red', linestyle='--', label=f'Observed score = {observed_score:.2f}')
    ax_perm.set_xlabel("Permutation Accuracy")
    ax_perm.set_ylabel("Count")
    ax_perm.set_title(f"{title_prefix} Permutation Test (p = {p_value:.4f})")
    ax_perm.legend()
    plt.tight_layout()
    plt.savefig(pjoin('figures', f'permutation_test_{title_prefix}.png'), dpi=600, bbox_inches='tight')
    plt.show()

def plot_bar_comparison(means, sems, group_names, title, filename):
    """
    Generates a bar plot with error bars for group comparisons.
    """
    # Create the 'figures' directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(5, 5))
    x_pos = np.arange(len(group_names))
    bar_width = 0.4
    
    colors = ['purple', 'orange']
    plt.bar(x_pos, means, yerr=sems, capsize=8, color=colors, width=bar_width)
    
    plt.xticks(x_pos, group_names)
    plt.ylabel('Accuracy scores')
    plt.xlabel('Groups')
    plt.title(title)
    plt.ylim(0, 1) # Accuracy scores are between 0 and 1
    
    plt.tight_layout()
    plt.savefig(pjoin('figures', filename), dpi=300, bbox_inches='tight')
    plt.show()

# Helper for collecting all predictions for confusion matrix
def get_all_predictions(g1_data, g2_data, all_subject_labels):
    loo = LeaveOneOut()
    X1_flat = [d.flatten() for d in g1_data]
    X2_flat = [d.flatten() for d in g2_data]
    true_all = []
    pred_all = []

    for train_idx1, test_idx1 in loo.split(X1_flat):
        X1_train, X1_test = np.array(X1_flat)[train_idx1], np.array(X1_flat)[test_idx1]
        y1_train, y1_test = np.array(labels_control)[train_idx1], np.array(labels_control)[test_idx1]
        
        for train_idx2, test_idx2 in loo.split(X2_flat):
            X2_train, X2_test = np.array(X2_flat)[train_idx2], np.array(X2_flat)[test_idx2]
            y2_train, y2_test = np.array(labels_als)[train_idx2], np.array(labels_als)[test_idx2]

            X_train = np.concatenate((X1_train, X2_train))
            y_train = np.concatenate((y1_train, y2_train))
            X_test = np.concatenate((X1_test, X2_test))
            y_test = np.concatenate((y1_test, y2_test))

            classifier = svm.SVC(kernel='linear')
            classifier.fit(X_train, y_train)
            predicted_labels = classifier.predict(X_test)
            
            true_all.extend(y_test)
            pred_all.extend(predicted_labels)
    return true_all, pred_all, accuracy_score(true_all, pred_all)


# --- Main Analysis Workflow ---
if __name__ == "__main__":
    print("--- Starting ALS fMRI Data Analysis ---")

    # --- Analysis for 'First Affected' Body Parts ---
    print("\n## Analyzing 'First Affected' Body Parts")
    
    # Split the preprocessed 'first affected' data into two runs
    # `arrs_con_first` and `arrs_als_first` contain data of shape (1500, 302)
    # They are split into (1500, 151) and (1500, 151)
    control_first_r1, control_first_r2 = split_data_into_runs(arrs_con_first, SPLIT_POINT_FULL_DATA)
    als_first_r1, als_first_r2 = split_data_into_runs(arrs_als_first, SPLIT_POINT_FULL_DATA)

    # Run SRM projection for 'first affected' data
    K_FEATURES = 10 # Number of features for RSRM
    run_crossval_srm_projection_corrected(
        control_first_r1, control_first_r2,
        als_first_r1, als_first_r2,
        K_FEATURES, "first_affected_data", n_iterations=30 # 'vox' in original corresponds to output_tag here
    )

    # Load projected data for 'first affected'
    # The saved structure from `run_crossval_srm_projection_corrected` is:
    # `projected_data=[[control_proj_R1_test_R2_list], [control_proj_R2_test_R1_list]]` for control file
    # `projected_data=[[als_proj_R1_test_R2_list], [als_proj_R2_test_R1_list]]` for als file

    # Load control projections
    with np.load(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_first_affected_data_0_{K_FEATURES}.npz'), allow_pickle=True) as data:
        projected_data_control_first = data['projected_data']
    
    # Load ALS projections
    with np.load(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_first_affected_data_1_{K_FEATURES}.npz'), allow_pickle=True) as data:
        projected_data_als_first = data['projected_data']

    # shared_all_y1 (Control, training on run 1, test on run 2)
    shared_control_first_r1_proj = projected_data_control_first[0] 
    # shared_all_y2 (Control, training on run 2, test on run 1)
    shared_control_first_r2_proj = projected_data_control_first[1]

    # shared_all_o1 (ALS, training on run 1, test on run 2)
    shared_als_first_r1_proj = projected_data_als_first[0]
    # shared_all_o2 (ALS, training on run 2, test on run 1)
    shared_als_first_r2_proj = projected_data_als_first[1]

    # Classify 'first affected' data using the relevant projected runs
    print("\n--- Classification for First Affected Data (Training on Run 1, Testing on Run 2) ---")
    scores_first_r1, mean_first_r1, std_first_r1, sem_first_r1, \
    cv_score_first_r1, perm_scores_first_r1, pval_first_r1 = \
        loo_classify_balanced_combination(shared_control_first_r1_proj, shared_als_first_r1_proj, all_labels)
    
    print("\n--- Classification for First Affected Data (Training on Run 2, Testing on Run 1) ---")
    scores_first_r2, mean_first_r2, std_first_r2, sem_first_r2, \
    cv_score_first_r2, perm_scores_first_r2, pval_first_r2 = \
        loo_classify_balanced_combination(shared_control_first_r2_proj, shared_als_first_r2_proj, all_labels)
    
    # Combined metrics for first affected
    # Combine scores from both runs
    all_first_scores_flattened = np.array(scores_first_r1).flatten().tolist() + np.array(scores_first_r2).flatten().tolist()
    combined_mean_first = np.mean(all_first_scores_flattened)
    combined_std_first = np.std(all_first_scores_flattened)
    combined_sem_first = combined_std_first / np.sqrt(len(all_first_scores_flattened))
    print(f"\nFirst Affected Combined Mean � SEM: {combined_mean_first:.4f} � {combined_sem_first:.4f}")

    # For confusion matrix and permutation plot, we typically want one set of true/predicted labels.
    # We'll use the results from the first classification run (training on R1, testing on R2) for this.
    true_labels_first, pred_labels_first, _ = get_all_predictions(shared_control_first_r1_proj, shared_als_first_r1_proj, all_labels)
    plot_confusion_matrix_and_permutation_test(
        true_labels_first, pred_labels_first, cv_score_first_r1, perm_scores_first_r1, pval_first_r1,
        title_prefix="First_Affected"
    )

    # --- Analysis for 'Later Affected' Body Parts ---
    print("\n## Analyzing 'Later Affected' Body Parts")
    
    # Get onsets for all body parts for both groups to create 'cyclic' data
    onsets_control = get_onsets_randomruns(control_subjects, BASE_DIR, group_name='control')
    onsets_als = get_onsets_randomruns(als_subjects, BASE_DIR, group_name='als')

    # Arrange (synchronize) 'later affected' data based on onsets.
    # Apply z-scoring here  for these specific  variables.
    control_cyclic_later = arrange_segments_from_onsets(control_subjects, onsets_control, arrs_con_later, apply_zscore=True)
    als_cyclic_later = arrange_segments_from_onsets(als_subjects, onsets_als, arrs_als_later, apply_zscore=True)

    # Split the 'later affected' cyclic data into two runs

    
    control_later_r1, control_later_r2 = split_data_into_runs(control_cyclic_later, SPLIT_POINT_CYCLIC_DATA)
    als_later_r1, als_later_r2 = split_data_into_runs(als_cyclic_later, SPLIT_POINT_CYCLIC_DATA)

    # Run SRM projection for 'later affected' data
    run_crossval_srm_projection_corrected(
        control_later_r1, control_later_r2,
        als_later_r1, als_later_r2,
        K_FEATURES, "later_affected_data", n_iterations=20 # Using 20 iterations as in original code
    )

    # Load projected data for 'later affected'
    # Note: original code path had '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results/'
    # which is likely a local mount. Using `RESULTS_DIR` from config.
    with np.load(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_later_affected_data_0_{K_FEATURES}.npz'), allow_pickle=True) as data:
        projected_data_control_later = data['projected_data']
    with np.load(pjoin(RESULTS_DIR, f'group_projection_rSRM_permuted_cyclic_firstvs2_later_affected_data_1_{K_FEATURES}.npz'), allow_pickle=True) as data:
        projected_data_als_later = data['projected_data']

    # shared_all_y1 (Control, training on run 1, test on run 2)
    shared_control_later_r1_proj = projected_data_control_later[0]
    # shared_all_y2 (Control, training on run 2, test on run 1)
    shared_control_later_r2_proj = projected_data_control_later[1]

    # shared_all_o1 (ALS, training on run 1, test on run 2)
    shared_als_later_r1_proj = projected_data_als_later[0]
    # shared_all_o2 (ALS, training on run 2, test on run 1)
    shared_als_later_r2_proj = projected_data_als_later[1]

    # Classify 'later affected' data using the relevant projected runs
    print("\n--- Classification for Later Affected Data (Training on Run 1, Testing on Run 2) ---")
    scores_later_r1, mean_later_r1, std_later_r1, sem_later_r1, \
    cv_score_later_r1, perm_scores_later_r1, pval_later_r1 = \
        loo_classify_balanced_combination(shared_control_later_r1_proj, shared_als_later_r1_proj, all_labels)

    print("\n--- Classification for Later Affected Data (Training on Run 2, Testing on Run 1) ---")
    scores_later_r2, mean_later_r2, std_later_r2, sem_later_r2, \
    cv_score_later_r2, perm_scores_later_r2, pval_later_r2 = \
        loo_classify_balanced_combination(shared_control_later_r2_proj, shared_als_later_r2_proj, all_labels)

    # Combined metrics for later affected
    all_later_scores_flattened = np.array(scores_later_r1).flatten().tolist() + np.array(scores_later_r2).flatten().tolist()
    combined_mean_later = np.mean(all_later_scores_flattened)
    combined_std_later = np.std(all_later_scores_flattened)
    combined_sem_later = combined_std_later / np.sqrt(len(all_later_scores_flattened))
    print(f"\nLater Affected Combined Mean � SEM: {combined_mean_later:.4f} � {combined_sem_later:.4f}")

    true_labels_later, pred_labels_later, _ = get_all_predictions(shared_control_later_r1_proj, shared_als_later_r1_proj, all_labels)
    plot_confusion_matrix_and_permutation_test(
        true_labels_later, pred_labels_later, cv_score_later_r1, perm_scores_later_r1, pval_later_r1,
        title_prefix="Later_Affected"
    )

    # --- Comparison Plot ---
    print("\n## Generating Comparison Plot")
    comparison_means = [combined_mean_first, combined_mean_later]
    comparison_sems = [combined_sem_first, combined_sem_later]
    comparison_groups = ['First affected', 'Others'] # 'Others' refers to 'Later affected' here

    plot_bar_comparison(comparison_means, comparison_sems, comparison_groups,
                        'First Affected vs Other Body Parts Accuracy', 'first_vs_later_accuracy_comparison.png')

    # --- Statistical Test for Comparison ---
    print("\n## Performing Statistical Test for Group Comparison")
    # For statistical comparison, it's best to use the individual scores or a more robust method
    # than just the combined means. A paired t-test on the individual LOO scores is appropriate
    # if the samples are paired (e.g., same subjects contribute to both 'first' and 'later' scores).
    # Assuming `all_first_scores_flattened` and `all_later_scores_flattened` represent
    # paired observations for each subject's classification across the two conditions.
    # It's crucial that these lists are of the same length and correspond.
    
    from scipy.stats import ttest_rel
    
    # Ensure lists are of the same length for paired t-test
    min_len = min(len(all_first_scores_flattened), len(all_later_scores_flattened))
    
    t_statistic, p_value_ttest = ttest_rel(all_first_scores_flattened[:min_len], all_later_scores_flattened[:min_len])
    print(f"Paired t-test between 'First affected' and 'Later affected':")
    print(f"t({min_len-1}) = {t_statistic:.2f}, p = {p_value_ttest:.4f}")

    if p_value_ttest < 0.05:
        print("There is a significant difference between the 'First affected' and 'Later affected' scores.")
    else:
        print("There is no significant difference between the 'First affected' and 'Later affected' scores.")

    print("\n--- Analysis Complete ---")
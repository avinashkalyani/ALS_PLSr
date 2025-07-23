import os.path
import warnings
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal # Combined imports for scipy
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import nibabel as nib

# --- Define Paths ---
# Base directory for the project data
base_dir = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'
results_dir = os.path.join(base_dir, 'results')
# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Specific file paths
control_behav_path = os.path.join(base_dir, 'control_behavioral.csv')
als_behav_path_old = os.path.join(base_dir, 'ALS_behavioral.csv')
als_behav_path_new = os.path.join(base_dir, 'New_ALS_behave.csv')
timeseries_data_path = os.path.join(results_dir, 'timeseries_WB_ALS_new.npz')
ref_image_path = os.path.join(base_dir, 'ALS_functional_localizer_masks', 'wb_functional_HFC.nii.gz')

# --- Load Behavioral Data ---
# Load control and ALS behavioral data
df_control_behav = pd.read_csv(control_behav_path)
df_als_behav_old = pd.read_csv(als_behav_path_old)
df_als_behav_new = pd.read_csv(als_behav_path_new)

# Concatenate old and new ALS behavioral dataframes
df_ALS_behav = pd.concat([df_als_behav_old, df_als_behav_new], ignore_index=True)

# Extract subject IDs (though not directly used in the main PLS loop for indexing, kept for context)
subj_control = df_control_behav['Subject']
subj_ALS = df_ALS_behav['Subject']

# --- Helper Functions ---

def r2_score_simple_linear(y_true, y_pred):
    """
    Calculates a simple R-squared (R2) score.
    """
    y_mean = np.mean(y_true)
    ssr = np.sum((y_pred - y_mean) ** 2) # Sum of Squares Regression
    sst = np.sum((y_true - y_mean) ** 2) # Total Sum of Squares
    
    # Avoid division by zero
    if sst == 0:
        return 0.0 # Or np.nan, depending on desired behavior for constant y_true
    
    r2 = 1 - (ssr / sst)
    return r2

def sliding_window_view(arr, window_shape):
    """
    Creates a sliding window view of an array.
    Note: This function was defined but not used in the provided script's main flow.
    """
    strides = arr.strides + arr.strides
    return np.lib.stride_tricks.as_strided(arr, shape=(*arr.shape[:-1], *window_shape), strides=strides)

def weights_plot(num_voxels, pls_model, test_idx, all_xyz_coords):
    """
    Reshapes PLS weights, maps them to 3D coordinates, and saves as a NIfTI image.
    """
    # Extract and reshape LV1 weights
    weights = pls_model.x_weights_[:, 0].reshape(num_voxels, -1)
    weights_mean = np.mean(weights, axis=1)

    # Combine XYZ coordinates with mean weights
    # Assuming all_xyz_coords[0] contains the (x, y, z) coordinates for each voxel
    weights_3D_data = np.concatenate([all_xyz_coords[0], weights_mean.reshape(-1, 1)], axis=1)

    # Extract coordinates and intensity values
    x_coords = weights_3D_data[:, 0].astype(int)
    y_coords = weights_3D_data[:, 1].astype(int)
    z_coords = weights_3D_data[:, 2].astype(int)
    intensity_values = weights_3D_data[:, 3]

    # Create a 3D volume
    max_x, max_y, max_z = np.max(x_coords), np.max(y_coords), np.max(z_coords)
    volume = np.zeros((max_x + 1, max_y + 1, max_z + 1))
    volume[x_coords, y_coords, z_coords] = intensity_values

    # Load reference image for affine
    reference_image = nib.load(ref_image_path)
    # Create NIfTI image from the volume and reference affine
    nifti_image = nib.Nifti1Image(volume, affine=reference_image.affine)

    # Define output path and save the NIfTI image
    out_path = os.path.join(results_dir, f'weights_rms_KS_BPF_{test_idx}_wnew_ALS.nii.gz')
    nib.save(nifti_image, out_path)
    print(f"Saved weights NIfTI for test_index {test_idx} to {out_path}")

# --- Load and Preprocess Timeseries Data ---
print(f"Loading timeseries data from {timeseries_data_path}...")
with np.load(timeseries_data_path, allow_pickle=True) as data:
    all_arrs = data['timeseries_data'] # Timeseries data for all subjects/voxels
    all_xyz = data['xyz_data'] # XYZ coordinates for voxels

print(f"Timeseries data shape: {all_arrs.shape}")

# Define bandpass filter parameters
low_cutoff_freq = 0.008  # Hz
high_cutoff_freq = 0.08  # Hz
sampling_freq = 0.50     # Hz (TR = 2s, so 1/2 = 0.5Hz)

nyquist_freq = 0.5 * sampling_freq
low_cutoff_norm = low_cutoff_freq / nyquist_freq
high_cutoff_norm = high_cutoff_freq / nyquist_freq

print(f"Applying bandpass filter ({low_cutoff_freq}-{high_cutoff_freq} Hz)...")
filtered_arrs = []
for arr in all_arrs:
    # Apply 4th order Butterworth bandpass filter
    b, a = signal.butter(4, [low_cutoff_norm, high_cutoff_norm], btype='band')
    filtered_arr = signal.filtfilt(b, a, arr, axis=-1) # Apply filter along time axis
    filtered_arrs.append(filtered_arr)
filtered_arrs = np.array(filtered_arrs)
print(f"Filtered timeseries data shape: {filtered_arrs.shape}")

# --- Prepare Behavioral (Y) Data ---
# Select relevant ALSFRS-R columns
alsfrs_columns = [
    'ALSFRS 3 Swallow', 'ALSFRS 2 Saliva', 'ALSFRS 1 Language',
    'ALSFRS 4 Handwriting', 'ALSFRS 5a cutting food',
    'ALSFRS 8 walking', 'ALSFRS 9 climbing stairs',
    'ALSFRS 10 shortness of breath', 'ALSFRS 11 orthopnea',
    'ALSFRS 12 respiratory failure'
]
df_ALSFRS = df_ALS_behav[alsfrs_columns].copy()

# Correct the indexing for removing rows to ensure alignment with fMRI data

# Re-aligning based on the final shape of X_data.
if filtered_arrs.shape[0] != df_ALSFRS.shape[0]:
    print(f"Warning: Number of subjects in fMRI data ({filtered_arrs.shape[0]}) "
          f"does not match behavioral data ({df_ALSFRS.shape[0]}). Please check alignment.")
    # For now, we'll truncate behavioral data to match fMRI if it's longer
    if df_ALSFRS.shape[0] > filtered_arrs.shape[0]:
        df_ALSFRS = df_ALSFRS.iloc[:filtered_arrs.shape[0]]
        labels = df_ALS_behav['King Stage'].iloc[:filtered_arrs.shape[0]].values # Align King Stage too
    else:
        # Handle case where fMRI data might be missing subjects from behavioral
        # This needs careful subject ID matching, which isn't explicit here.
        # For simplicity, assuming fMRI is the master list for now.
        raise ValueError("fMRI data has more subjects than behavioral data or subjects are misaligned. "
                         "Explicit subject ID matching is recommended.")
else:
    labels = df_ALS_behav['King Stage'].values # Get King Stage labels

y = np.nan_to_num(stats.zscore(df_ALSFRS.values, axis=0)) # Z-score and handle NaNs

# --- Prepare fMRI (X) Data ---
# Preprocess and reshape fMRI data for PLS
# `filtered_arrs` is the actual fMRI data to be used.
X = np.nan_to_num(stats.zscore(filtered_arrs, axis=1)) # Z-score along voxel dimension
n_voxels = X.shape[1] # Number of voxels

# Flatten the time dimension for PLS (subjects x (voxels * timepoints))

num_subjects = X.shape[0]
X_flat = X.reshape(num_subjects, -1) # Reshape to (subjects, voxels * timepoints)


# 14 subjects after deletion:
if num_subjects > 14: # Assuming 14 is the target count after previous filtering/deletion
    print(f"Warning: More than 14 subjects found ({num_subjects}). Removing the second-to-last subject to match original script's behavior.")
    # Consider replacing with explicit subject filtering if possible.
    X_flat = np.delete(X_flat, -2, axis=0)
    y = np.delete(y, -2, axis=0)
    labels = np.delete(labels, -2, axis=0)
    # Update num_subjects for subsequent checks
    num_subjects = X_flat.shape[0]

print(f"Final preprocessed X data shape: {X_flat.shape}")
print(f"Final preprocessed Y data shape: {y.shape}")
print(f"Final labels shape: {labels.shape}")


# --- Partial Least Squares (PLS) Regression ---
num_components = 2 # Number of PLS components
pls = PLSRegression(n_components=num_components)
loo = LeaveOneOut() # Leave-One-Out cross-validation

# Initialize lists to store results
mse_values = []
r2_values = []
x_scores_all = []       # X scores from training data
y_scores_all = []       # Y scores from training data
x_latent_test_all = []  # X latent variables from test subject
y_latent_test_all = []  # Y latent variables from test subject
labels_train_all = []   # Labels from training subjects
labels_test_all = []    # Labels from test subject

print("\n--- Starting Leave-One-Out PLS Cross-Validation ---")
# Iterate through each sample for Leave-One-Out cross-validation
for i, (train_index, test_index) in enumerate(loo.split(X_flat)):
    X_train, X_test = X_flat[train_index], X_flat[test_index]
    y_train, y_test = y[train_index], y[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Fit the PLS model on the training data
    pls.fit(X_train, y_train)
    
    # Predict behavioral data on the test set
    y_pred = pls.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

    # Get the latent variables (scores) for X and Y from the training fit
    x_latent_variables_train = pls.x_scores_
    y_latent_variables_train = pls.y_scores_
    
    # Transform the test subject's data to get their latent variables
    x_latent_test, y_latent_test = pls.transform(X_test, y_test)
    
    # Save voxel weights (LV1) as a NIfTI image for the current test fold
    # This will save a brain map of the LV1 weights from the model fitted on the current fold's training data.
    # The `test_index` refers to the index in the original dataset `X_flat`.
    # Using `i` for the fold number to make file names sequential if needed.
    weights_plot(n_voxels, pls, test_index[0], all_xyz) # test_index[0] is the original index of the test subject

    # Append scores and labels to lists
    x_scores_all.append(x_latent_variables_train[:, 0]) # LV1 for training subjects
    y_scores_all.append(y_latent_variables_train[:, 0]) # LV1 for training subjects
    x_latent_test_all.append(x_latent_test)             # All LVs for test subject
    y_latent_test_all.append(y_latent_test)             # All LVs for test subject
    labels_train_all.append(labels_train)
    labels_test_all.append(labels_test)

    # Calculate R-squared
    if np.var(y_test) != 0:
        r2 = r2_score_simple_linear(y_test, y_pred)
        r2_values.append(r2)
        print(f"  Fold {i+1}: R2 = {r2:.4f}, MSE = {mse:.4f}")
    else:
        print(f"  Fold {i+1}: Variance of y_test is zero, skipping R2. MSE = {mse:.4f}")

# Convert lists to NumPy arrays
x_scores_all = np.array(x_scores_all)
y_scores_all = np.array(y_scores_all)
x_latent_test_all = np.array(x_latent_test_all)
y_latent_test_all = np.array(y_latent_test_all)
labels_train_all = np.array(labels_train_all)
labels_test_all = np.array(labels_test_all)

# --- Summarize PLS Results ---
mean_mse = np.mean(mse_values)
sem_mse = stats.sem(mse_values)
print(f"\nOverall Mean MSE: {mean_mse:.4f} (SEM: {sem_mse:.4f})")
print(f"Overall Mean R2: {np.mean(r2_values):.4f}")

# Plot histogram of MSE values
plt.figure(figsize=(6, 4))
plt.hist(mse_values, bins=5, edgecolor='black')
plt.title('Distribution of MSE Values Across LOO Folds')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'mse_distribution.png'))
plt.close()

# --- Plot PLS Scores ---
print("\nGenerating PLS scores plot...")
# Dynamically determine rows for subplots
num_plots = len(mse_values)
cols = 7 # Number of columns for subplots
rows = (num_plots + cols - 1) // cols # Calculate required rows
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axs = axs.flat # Flatten array for easy iteration

for i in range(num_plots):
    ax = axs[i]
    # Plot training data scores (LV1)
    ax.scatter(x_scores_all[i], y_scores_all[i], c=labels_train_all[i], cmap='viridis', alpha=0.9, vmin=1, vmax=3)
    
    # Plot test subject's score (LV1)
    ax.scatter(x_latent_test_all[i][:, 0], y_latent_test_all[i][:, 0],
               edgecolor='red', facecolor='none', marker='o', s=100, linewidth=2)
    ax.scatter(x_latent_test_all[i][:, 0], y_latent_test_all[i][:, 0],
               c=labels_test_all[i], marker='o', cmap='viridis', vmin=1, vmax=3, s=40)
    
    ax.set_xlabel('fMRI Score (LV 1)')
    ax.set_ylabel('Behavioral Score (LV 1)')
    ax.set_title(f'Test Split {i + 1}, MSE: {mse_values[i]:.2f}')

# Turn off any unused subplots
for j in range(num_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'scores_KingStage_newsub_func_activity.png'), dpi=600)
plt.close()
print(f"PLS scores plot saved to {os.path.join(results_dir, 'scores_KingStage_newsub_func_activity.png')}")

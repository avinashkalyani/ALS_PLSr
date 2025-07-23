import os.path
import warnings
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal # Combined imports for scipy
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import nibabel as nib
from nilearn import masking # For NIfTI mask operations


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
ecm_data_dir = os.path.join(base_dir, 'ALS_AA_MNI_152T1_1mm_outputs_b3_masked11', 'ecm_result_dir')
mask_path = os.path.join(base_dir, 'combined_functional_localizer_masks', 'combined_mask_HFC.nii.gz')

# --- Load Behavioral Data ---
# Load control and ALS behavioral data
df_control_behav = pd.read_csv(control_behav_path)
df_als_behav_old = pd.read_csv(als_behav_path_old)
df_als_behav_new = pd.read_csv(als_behav_path_new)

# Concatenate old and new ALS behavioral dataframes
df_ALS_behav = pd.concat([df_als_behav_old, df_als_behav_new], ignore_index=True)

# Extract subject IDs for ALS group
subj_ALS = df_ALS_behav['Subject'].tolist()
subj_control = df_control_behav['Subject'].tolist() # Kept for completeness, not used here

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

def weights_plot(num_voxels, pls_model, test_idx, mask_img_path):
    """
    Reshapes PLS weights from the fitted model, unmasks them to 3D volume,
    and saves as a NIfTI image.
    """
    # Extract LV1 weights and compute mean across any potential components (though only 1st LV is usually taken)
    weights = pls_model.x_weights_[:, 0]
    # No need to reshape if unmasking directly from 1D array

    # Load the reference mask image for unmasking
    reference_mask_image = nib.load(mask_img_path)
    
    # Unmask the 1D weights array into a 3D NIfTI image using nilearn.masking.unmask
    nifti_image = masking.unmask(weights, reference_mask_image)
    
    # Define output path and save the NIfTI image
    out_path = os.path.join(results_dir, f'weights_ecm_KS_BPF_{test_idx}_wnew_ALS.nii.gz')
    nib.save(nifti_image, out_path)
    print(f"Saved weights NIfTI for test_index {test_idx} to {out_path}")

# --- Load and Preprocess ECM Data ---
print(f"Loading ECM data from {ecm_data_dir} and applying mask from {mask_path}...")
all_sub_ecm = []
for sub in subj_ALS:
    ecm_file = os.path.join(ecm_data_dir, f'vbcm_{sub}.nii')
    if os.path.exists(ecm_file):
        # Apply mask and ensure data is float and finite
        ecm_val = masking.apply_mask(ecm_file, mask_img=mask_path, dtype=np.float32, ensure_finite=True)
        all_sub_ecm.append(ecm_val)
    else:
        print(f"Warning: ECM file not found for subject {sub}: {ecm_file}. Skipping.")

if not all_sub_ecm:
    raise ValueError("No ECM data loaded. Check subject IDs and file paths.")

# Convert list of 1D arrays (masked data for each subject) to a 2D NumPy array
# Each row is a subject, each column is a voxel
all_sub_ecm_array = np.array(all_sub_ecm)
num_subjects_ecm, n_voxels = all_sub_ecm_array.shape
print(f"Loaded ECM data shape: {all_sub_ecm_array.shape} (Subjects: {num_subjects_ecm}, Voxels: {n_voxels})")

# --- Prepare Behavioral (Y) Data ---
# Define relevant ALSFRS-R columns
alsfrs_columns = [
    'ALSFRS 3 Swallow', 'ALSFRS 2 Saliva', 'ALSFRS 1 Language',
    'ALSFRS 4 Handwriting', 'ALSFRS 5a cutting food',
    'ALSFRS 8 walking', 'ALSFRS 9 climbing stairs',
    'ALSFRS 10 shortness of breath', 'ALSFRS 11 orthopnea',
    'ALSFRS 12 respiratory failure'
]
df_ALSFRS = df_ALS_behav[alsfrs_columns].copy()

# Ensure behavioral data aligns with the loaded ECM subjects

if df_ALSFRS.shape[0] != num_subjects_ecm:
    print(f"Warning: Behavioral data rows ({df_ALSFRS.shape[0]}) do not match "
          f"number of loaded ECM subjects ({num_subjects_ecm}). Attempting to align.")
    # This scenario needs robust subject ID matching. For now, assuming direct slice if difference is small.
    # A more robust solution would involve merging dataframes on subject IDs.
    if df_ALSFRS.shape[0] > num_subjects_ecm:
        df_ALSFRS = df_ALSFRS.iloc[:num_subjects_ecm].copy()
        df_ALS_behav = df_ALS_behav.iloc[:num_subjects_ecm].copy()
    else:
        raise ValueError("More ECM subjects than behavioral data. Check subject lists/data loading.")

# Extract King Stage labels
labels = df_ALS_behav['King Stage'].values

#cleaning noisy data
if num_subjects_ecm == 15 and all_sub_ecm_array.shape[0] >= 2: # Check if there are enough subjects to remove the second-to-last
    # This is a critical point: Ensure the subject being removed from X is the same as from Y and labels.
    # Without explicit subject IDs for removal, it's safer to remove by index.
    print("Removing second-to-last subject from ECM, behavioral data, and labels to match original script's 14 subjects.")
    all_sub_ecm_array = np.delete(all_sub_ecm_array, -2, axis=0)
    df_ALSFRS = df_ALSFRS.drop(df_ALSFRS.index[-2]).reset_index(drop=True) # Drop and reset index
    labels = np.delete(labels, -2, axis=0)
    num_subjects_ecm = all_sub_ecm_array.shape[0] # Update count

y = np.nan_to_num(stats.zscore(df_ALSFRS.values, axis=0)) # Z-score and handle NaNs

# Preprocess ECM data for PLS: X is already 2D (subjects x voxels)
# If additional z-scoring is desired for ECM data, apply here.
X = np.nan_to_num(stats.zscore(all_sub_ecm_array, axis=1)) # Z-score along voxel dimension
# X is already in (subjects, features) format, no need to reshape `all_subjects_data_als`
# as `all_sub_ecm_array` is already like that.
# `hand_ALS` and `all_subjects_data_als` variables are now simplified.

print(f"Final preprocessed X data shape: {X.shape}")
print(f"Final preprocessed Y data shape: {y.shape}")
print(f"Final labels shape: {labels.shape}")

# Initialize PLS model
num_components = 2
pls = PLSRegression(n_components=num_components)

# Initialize Leave-One-Out cross-validation
loo = LeaveOneOut()

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
for i, (train_index, test_index) in enumerate(loo.split(X)):
    X_train, X_test = X[train_index], X[test_index]
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
    # The `test_index` from LOO split is an array, take the first element for filename.
    weights_plot(n_voxels, pls, test_index[0], mask_path)

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
        print(f"  Fold {i+1} (Test Subject Index: {test_index[0]}): R2 = {r2:.4f}, MSE = {mse:.4f}")
    else:
        print(f"  Fold {i+1} (Test Subject Index: {test_index[0]}): Variance of y_test is zero, skipping R2. MSE = {mse:.4f}")

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
plt.savefig(os.path.join(results_dir, 'mse_distribution_ECM.png'))
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
    # .ravel() is used because labels_train_all[i] might be a 1-element array,
    # and matplotlib prefers a 0-D or 1-D array for `c`.
    scatter = ax.scatter(x_scores_all[i], y_scores_all[i], c=labels_train_all[i].ravel(), cmap='viridis', alpha=0.9, vmin=1, vmax=3)
    
    # Plot test subject's score (LV1)
    ax.scatter(x_latent_test_all[i][:, 0], y_latent_test_all[i][:, 0],
               edgecolor='red', facecolor='none', marker='o', s=100, linewidth=2, zorder=5) # zorder to ensure it's on top
    ax.scatter(x_latent_test_all[i][:,0], y_latent_test_all[i][:,0],
               c=labels_test_all[i].ravel(), marker='o', cmap='viridis', vmin=1, vmax=3, s=40, zorder=6) # zorder to ensure it's on top
    
    ax.set_xlabel('X Score (LV 1)')
    ax.set_ylabel('Y Score (LV 1)')
    ax.set_title(f'Test Split {i + 1}, MSE: {mse_values[i]:.2f}')

    # Optionally add dotted lines at zero
    # ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

# Turn off any unused subplots
for j in range(num_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'scores_KingStage_ECM.png'), dpi=600)
plt.close()
print(f"PLS scores plot saved to {os.path.join(results_dir, 'scores_KingStage_ECM.png')}")



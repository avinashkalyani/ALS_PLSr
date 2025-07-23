import os
import warnings
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, pearsonr, ttest_rel # Added pearsonr and ttest_rel here
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from nilearn import masking
from nilearn import plotting # Added for brain plotting


# --- Configuration and Path Definitions ---
BASE_DIR = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Create results and figures directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define file paths
CONTROL_BEHAV_PATH = os.path.join(BASE_DIR, 'control_behavioral.csv')
ALS_BEHAV_PATH_OLD = os.path.join(BASE_DIR, 'ALS_behavioral.csv')
ALS_BEHAV_PATH_NEW = os.path.join(BASE_DIR, 'New_ALS_behave.csv')
QSM_BASE_DIR = os.path.join(BASE_DIR, 'QSM_MNI_outputs_re_registered')
QSM_ALS_DIR = os.path.join(QSM_BASE_DIR, 'ALS_QSM_MNI2')
MASK_PATH = os.path.join(BASE_DIR, 'ALS_functional_localizer_masks', 'wb_functional_HFC.nii.gz')
REF_IMAGE_PATH = os.path.join(BASE_DIR, 'ALS_functional_localizer_masks', 'wb_functional_HFC.nii.gz')

# ROI Masks for analysis
HAND_MASK_PATH = os.path.join(BASE_DIR, "ALS_functional_localizer_masks", "roi_vbm", "largest_cluster_mask_hand_merged.nii")
FOOT_MASK_PATH = os.path.join(BASE_DIR, "ALS_functional_localizer_masks", "roi_vbm", "largest_cluster_mask_foot_merged.nii")
TONGUE_MASK_PATH = os.path.join(BASE_DIR, "ALS_functional_localizer_masks", "roi_vbm", "largest_cluster_mask_tongue_merged.nii")

# ALS subjects with QSM data (from your original script)
ALS_QSM_SUBJECTS = ["dl35", "na78", 'pn47', 'rs25', 'vj48', 'bo91', 'nv39', 'bo58']

# --- Helper Functions ---

def load_behavioral_data(old_path, new_path, qsm_subjects):
    """
    Loads and combines ALS behavioral data, then filters for subjects
    present in the QSM dataset.
    """
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)
    
    # Combine and remove duplicates, if any
    df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['Subject'])
    
    # Filter for subjects that have QSM data
    df_filtered = df_combined[df_combined['Subject'].isin(qsm_subjects)].reset_index(drop=True)
    
    # Define columns to include for ALSFRS-R scores
    alsfrs_columns_to_include = [
        'ALSFRS 3 Swallow', 'ALSFRS 2 Saliva', 'ALSFRS 1 Language',
        'ALSFRS 4 Handwriting', 'ALSFRS 5a cutting food', 'ALSFRS 8 walking',
        'ALSFRS 9 climbing stairs', 'ALSFRS 10 shortness of breath',
        'ALSFRS 11 orthopnea', 'ALSFRS 12 respiratory failure'
    ]
    
    # Select and return the relevant ALSFRS-R scores
    return df_filtered[['Subject'] + alsfrs_columns_to_include], df_filtered['King Stage'] # Return King Stage as well

def load_qsm_data(qsm_dir, mask_path, subjects):
    """
    Loads QSM data for a list of subjects, applies a mask, and returns
    a list of masked QSM arrays.
    """
    mask_img = nib.load(mask_path)
    all_sub_qsm_data = []
    for sub in subjects:
        qsm_filepath = os.path.join(qsm_dir, f'{sub}/QSM_re_registered_to_MNI.nii.gz')
        if os.path.exists(qsm_filepath):
            qsm_val = masking.apply_mask(qsm_filepath, mask_img, dtype='f', ensure_finite=True)
            all_sub_qsm_data.append(qsm_val)
        else:
            print(f"Warning: QSM file not found for {sub} at {qsm_filepath}. Skipping.")
    return np.array(all_sub_qsm_data)

def r2_score_custom(y_true, y_pred):
    """Calculates R-squared (R2) score."""
    y_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    
    if total_sum_squares == 0:
        return np.nan # Avoid division by zero if y_true is constant
        
    return 1 - (residual_sum_squares / total_sum_squares)

def save_voxel_weights_as_nifti(weights_mean, ref_image_path, output_filepath):
    """
    Unmasks mean voxel weights and saves them as a NIfTI image.
    """
    ref_image = nib.load(ref_image_path)
    nifti_image = masking.unmask(weights_mean, ref_image)
    nib.save(nifti_image, output_filepath)
    print(f"Saved PLS weights to {output_filepath}")

def extract_mean_weight_from_roi(stat_img_path, roi_mask_path, ref_image_path):
    """
    Extracts the mean weight within a specified ROI, ensuring the ROI mask
    is aligned with the reference image.
    """
    try:
        stat_img_data = nib.load(stat_img_path).get_fdata()
        roi_mask_data = nib.load(roi_mask_path).get_fdata()
        ref_img_data = nib.load(ref_image_path).get_fdata()

        # Ensure masks are boolean and same shape as stat_img
        
        combined_mask = np.logical_and(roi_mask_data > 0, ref_img_data > 0)
        
        # Apply the combined mask
        weights = stat_img_data[combined_mask]
        
        return np.mean(weights) if weights.size > 0 else 0.0
    except Exception as e:
        print(f"Error extracting mean weight for {stat_img_path} with {roi_mask_path}: {e}")
        return 0.0 # Return 0.0 or NaN in case of error

# --- Main Analysis Functions ---

def perform_pls_analysis(X_data, Y_data, labels, n_components, n_voxels, qsm_subjects, ref_image_path):
    """
    Performs Leave-One-Out PLS regression and saves results.
    """
    pls = PLSRegression(n_components=n_components)
    loo = LeaveOneOut()

    mse_values = []
    r2_values = []
    x_latent_scores_all = [] # Training set X latent scores
    y_latent_scores_all = [] # Training set Y latent scores
    x_latent_test_all = []   # Test set X latent scores
    y_latent_test_all = []   # Test set Y latent scores
    test_labels_all = []     # Actual labels of test subjects

    print("\n--- Starting PLS Leave-One-Out Cross-Validation ---")
    for i, (train_index, test_index) in enumerate(loo.split(X_data)):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        print(f"Processing LOO fold {i+1}/{len(qsm_subjects)}")
        
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

        # Save weights as NIfTI for each fold (as per original code)
        weights_mean = pls.x_weights_[:, 0].reshape(n_voxels, -1).mean(axis=1) # LV1 weights
        output_filepath = os.path.join(RESULTS_DIR, f'weights_QSM_KS_{qsm_subjects[test_index[0]]}_LV1.nii.gz')
        save_voxel_weights_as_nifti(weights_mean, ref_image_path, output_filepath)

        # Get latent variables for training data (to plot scatter)
        x_latent_train, y_latent_train = pls.transform(X_train, y_train)
        x_latent_scores_all.append(x_latent_train[:, 0])
        y_latent_scores_all.append(y_latent_train[:, 0])

        # Get latent variables for test data
        x_latent_test, y_latent_test = pls.transform(X_test, y_test)
        x_latent_test_all.append(x_latent_test[0, :]) # Take all components for test subject
        y_latent_test_all.append(y_latent_test[0, :])
        
        test_labels_all.append(labels_test[0])

        # Calculate R-squared
        if np.var(y_test) != 0:
            r2 = r2_score_custom(y_test, y_pred)
            r2_values.append(r2)
            # print(f"  R2 for test subject {qsm_subjects[test_index[0]]}: {r2:.4f}")
        else:
            print(f"  Variance of y_test is zero for subject {qsm_subjects[test_index[0]]}, skipping R2 calculation.")
    
    print(f"\nMean MSE: {np.mean(mse_values):.4f} � {stats.sem(mse_values):.4f}")
    print(f"Mean R2: {np.mean(r2_values):.4f} � {stats.sem(r2_values):.4f}")

    return {
        'pls_model_final': pls, # Keep the last fitted model for full data loadings
        'mse_values': mse_values,
        'r2_values': r2_values,
        'x_latent_scores_all': np.array(x_latent_scores_all),
        'y_latent_scores_all': np.array(y_latent_scores_all),
        'x_latent_test_all': np.array(x_latent_test_all),
        'y_latent_test_all': np.array(y_latent_test_all),
        'test_labels_all': np.array(test_labels_all)
    }

def plot_pls_scores_and_mse(results, qsm_subjects, output_path):
    """
    Plots the PLS latent variable scores for training and test data,
    including MSE for each fold.
    """
    num_folds = len(results['mse_values'])
    fig, axs = plt.subplots(int(np.ceil(num_folds / 2)), 2, figsize=(10, num_folds * 2.5))
    axs = axs.flat

    for i in range(num_folds):
        ax = axs[i]
        # Plot training data
        scatter = ax.scatter(results['x_latent_scores_all'][i], results['y_latent_scores_all'][i],
                             c=results['test_labels_all'][i], cmap='viridis', alpha=0.9, vmin=1, vmax=3) # Use test labels as a proxy for color
        
        # Plot test data
        ax.scatter(results['x_latent_test_all'][i, 0], results['y_latent_test_all'][i, 0],
                   edgecolor='red', facecolor='none', marker='o', s=100, linewidth=2)
        ax.scatter(results['x_latent_test_all'][i, 0], results['y_latent_test_all'][i, 0],
                   c=[results['test_labels_all'][i]], cmap='viridis', vmin=1, vmax=3, s=40)
        
        ax.set_xlabel('X Score (LV 1)')
        ax.set_ylabel('Y Score (LV 1)')
        ax.set_title(f'Test Subject: {qsm_subjects[i]}, Stage: {int(results["test_labels_all"][i])}, MSE: {results["mse_values"][i]:.2f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"PLS scores plot saved to {output_path}")

def plot_behavioral_loadings(pls_model, behav_columns, output_path):
    """
    Plots the PLS loadings on behavioral features for LV1 and LV2.
    """
    lv1_weights = pls_model.y_weights_[:, 0]
    lv2_weights = pls_model.y_weights_[:, 1]

    loading_df = pd.DataFrame({
        'Behavior': behav_columns,
        'LV1': lv1_weights,
        'LV2': lv2_weights
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(behav_columns))

    ax.bar(index, loading_df['LV1'], bar_width, label='LV1', color='skyblue')
    ax.bar(index + bar_width, loading_df['LV2'], bar_width, label='LV2', color='orange')

    ax.set_xlabel('Behavioral Features')
    ax.set_ylabel('Loading Weight')
    ax.set_title('PLSR Loadings on Behavioral Features')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(behav_columns, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Behavioral loadings plot saved to {output_path}")

def plot_subject_lv_projection(results, qsm_subjects, output_path, kings_stage_labels):
    """
    Plots each subject's projection in LV space (LV1 vs LV2 for fMRI data).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a mapping for King Stage to colors
    unique_stages = sorted(list(np.unique(kings_stage_labels)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
    stage_to_color = {stage: colors[i] for i, stage in enumerate(unique_stages)}

    for i in range(len(results['x_latent_test_all'])):
        subj_name = qsm_subjects[i]
        lv1_fmri = results['x_latent_test_all'][i, 0]
        lv2_fmri = results['x_latent_test_all'][i, 1]
        king_stage = int(kings_stage_labels[i]) # Ensure king stage is integer for indexing
        
        # Use the stage_to_color map
        color = stage_to_color[king_stage]
        
        ax.scatter(lv1_fmri, lv2_fmri, c=color, label=f'Stage {king_stage}' if subj_name == qsm_subjects[0] else '', alpha=0.8) # Label once
        ax.text(lv1_fmri, lv2_fmri, subj_name, fontsize=8, ha='right', va='bottom')

    # Create a single legend from the stage_to_color map
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=stage_to_color[stage], markersize=10, label=f'Stage {int(stage)}') 
               for stage in unique_stages]
    ax.legend(handles=handles, title="King Stage")

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('fMRI Score (LV1)')
    ax.set_ylabel('fMRI Score (LV2)')
    ax.set_title('Subject Projection in LV Space (QSM)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Subject LV projection plot saved to {output_path}")

def project_lv_to_brain(pls_model, qsm_data, mask_img, subjects, results_dir):
    """
    Projects LV weights onto individual subject QSM data and saves
    the resulting brain images.
    """
    lv1_weights = pls_model.x_weights_[:, 0]
    lv2_weights = pls_model.x_weights_[:, 1]

    for i, sub in enumerate(subjects):
        subject_data = qsm_data[i] # Already masked data
        
        # Projection score = element-wise product of QSM data and LV weights
        
        # Original logic: LV_weights * (dot product of subject_data and LV_weights)
        # This will scale the LV_weights by a scalar projection value.
        proj_lv1_map = lv1_weights * np.dot(subject_data, lv1_weights)
        proj_lv2_map = lv2_weights * np.dot(subject_data, lv2_weights)

        img_lv1 = masking.unmask(proj_lv1_map, mask_img)
        img_lv2 = masking.unmask(proj_lv2_map, mask_img)

        nib.save(img_lv1, os.path.join(results_dir, f'LV1_brain_projection_{sub}.nii.gz'))
        nib.save(img_lv2, os.path.join(results_dir, f'LV2_brain_projection_{sub}.nii.gz'))
        print(f"Saved LV1 & LV2 projections to brain space for {sub}")

def plot_brain_projections(subjects, results_dir, mask_path, fig_dir):
    """
    Plots LV1 and LV2 brain projections on a surface mesh.
    """
    mask_img = nib.load(mask_path)
    ref_image_for_plotting = nib.load(mask_path) # Use mask as reference for plotting affine
    
    surf_mesh = 'fsaverage5'
    # Adjusted threshold as it was very low in original, possibly indicating it needed to be unmasked more appropriately
    # or the projections are very small. Setting a reasonable default.
    threshold = 0.0001 # This might need tuning based on your actual projected values

    for sub in subjects:
        lv1_path = os.path.join(results_dir, f'LV1_brain_projection_{sub}.nii.gz')
        lv2_path = os.path.join(results_dir, f'LV2_brain_projection_{sub}.nii.gz')

        if not os.path.exists(lv1_path) or not os.path.exists(lv2_path):
            print(f"Skipping plotting for {sub}: projection NIfTI files not found.")
            continue

        img_lv1 = nib.load(lv1_path)
        img_lv2 = nib.load(lv2_path)

        # Plot LV1
        plotting.plot_img_on_surf(
            img_lv1,
            surf_mesh=surf_mesh,
            views=['lateral', 'medial'],
            title=f'LV1 Brain Projection: {sub}',
            colorbar=True,
            threshold=threshold
        )
        plt.savefig(os.path.join(fig_dir, f'LV1_brain_projection_surf_{sub}.png'))
        plt.close()

        # Plot LV2
        plotting.plot_img_on_surf(
            img_lv2,
            surf_mesh=surf_mesh,
            views=['lateral', 'medial'],
            title=f'LV2 Brain Projection: {sub}',
            colorbar=True,
            threshold=threshold
        )
        plt.savefig(os.path.join(fig_dir, f'LV2_brain_projection_surf_{sub}.png'))
        plt.close()
        print(f"Generated surface plots for {sub}")

def get_regional_lv_means(subjects, lv_paths_list, hand_mask, foot_mask, tongue_mask, ref_img_path):
    """
    Calculates and returns mean LV weights for Hand, Foot, and Tongue regions for each subject.
    """
    hand_means, foot_means, tongue_means = [], [], []
    for path in lv_paths_list:
        hand_means.append(extract_mean_weight_from_roi(path, hand_mask, ref_img_path))
        foot_means.append(extract_mean_weight_from_roi(path, foot_mask, ref_img_path))
        tongue_means.append(extract_mean_weight_from_roi(path, tongue_mask, ref_img_path))
    
    df_regional_means = pd.DataFrame({
        "Subject": subjects,
        "Hand": hand_means,
        "Foot": foot_means,
        "Tongue": tongue_means
    })
    return df_regional_means

def plot_per_subject_region_weights(lv1_df, lv2_df, output_path):
    """
    Plots per-subject region weights for LV1 and LV2.
    """
    n_subjects = len(lv1_df)
    subjects_labels = [f"Sub-{i+1}" for i in range(n_subjects)]

    x = np.arange(n_subjects)
    bar_width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # LV1 plot
    axs[0].bar(x - bar_width, lv1_df['Hand'], width=bar_width, label="Hand", color='red', alpha=0.7)
    axs[0].bar(x, lv1_df['Foot'], width=bar_width, label="Foot", color='green', alpha=0.7)
    axs[0].bar(x + bar_width, lv1_df['Tongue'], width=bar_width, label="Tongue", color='blue', alpha=0.7)
    axs[0].set_title("LV1 Region Weights")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(subjects_labels, rotation=45, ha='right')
    axs[0].set_ylabel("Mean Weight")
    axs[0].legend()

    # LV2 plot
    axs[1].bar(x - bar_width, lv2_df['Hand'], width=bar_width, label="Hand", color='red', alpha=0.7)
    axs[1].bar(x, lv2_df['Foot'], width=bar_width, label="Foot", color='green', alpha=0.7)
    axs[1].bar(x + bar_width, lv2_df['Tongue'], width=bar_width, label="Tongue", color='blue', alpha=0.7)
    axs[1].set_title("LV2 Region Weights")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(subjects_labels, rotation=45, ha='right')
    axs[1].legend()

    fig.suptitle("Per-Subject Region Weights for LV1 and LV2 (X as QSM)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Per-subject region weights plot saved to {output_path}")

def plot_average_region_weights(lv1_df, lv2_df, output_path):
    """
    Plots average region weights for LV1 and LV2 across all subjects.
    """
    labels = ["Hand", "Foot", "Tongue"]
    lv1_means = [lv1_df['Hand'].mean(), lv1_df['Foot'].mean(), lv1_df['Tongue'].mean()]
    lv2_means = [lv2_df['Hand'].mean(), lv2_df['Foot'].mean(), lv2_df['Tongue'].mean()]

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.bar(x - bar_width/2, lv1_means, width=bar_width, label='LV1', color='skyblue')
    ax.bar(x + bar_width/2, lv2_means, width=bar_width, label='LV2', color='orange')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Weight", fontsize=14, fontweight='bold')
    ax.set_xlabel("Region", fontsize=14, fontweight='bold')
    ax.set_title("Average Region Weights Across Subjects (X as QSM values)", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Average region weights plot saved to {output_path}")

def analyze_correlations_with_behavior(df_behav, lv_weights_df, als_qsm_subjects, results_dir):
    """
    Performs Spearman correlation analysis between LV region weights and behavioral scores.
    """
    features_mapping = {
        'Hand': ['ALSFRS 4 Handwriting', 'ALSFRS 5a cutting food'],
        'Foot': ['ALSFRS 8 walking', 'ALSFRS 9 climbing stairs'],
        'Tongue': ['ALSFRS 3 Swallow', 'ALSFRS 1 Language']
    }

    results = []

    for lv_label in ['LV1', 'LV2']:
        for region, behav_features in features_mapping.items():
            # Get the correct column name from the df_regional_means (e.g., Hand, Foot, Tongue)
            region_weights = lv_weights_df[f"{region}_{lv_label}"]
            
            for feat in behav_features:
                if feat in df_behav.columns:
                    # Align behavioral scores with the subjects present in QSM data
                    # Ensure behavioral scores match the length of region_weights (which should be len(ALS_QSM_SUBJECTS))
                    # df_behav is already filtered for ALS_QSM_SUBJECTS
                    behav_scores = df_behav[feat].values[:len(region_weights)] # Slice to match subjects if needed
                    
                    rho, pval = spearmanr(region_weights, behav_scores)
                    results.append({
                        'LV': lv_label,
                        'Region': region,
                        'Feature': feat,
                        'Spearman �': rho,
                        'p-value': pval
                    })
    
    df_stats = pd.DataFrame(results)
    df_stats = df_stats.sort_values('p-value').round(4)
    print("\n--- Spearman Correlation Results (QSM LV Weights vs. Behavioral Scores) ---")
    print(df_stats)
    df_stats.to_csv(os.path.join(results_dir, "QSM_LV_behavioral_correlations.csv"), index=False)
    print(f"Correlation results saved to {os.path.join(results_dir, 'QSM_LV_behavioral_correlations.csv')}")

    return df_stats

def compare_qsm_fmri_weights(df_qsm, df_fmri, results_dir):
    """
    Compares Z-scored QSM and fMRI regional LV weights using correlation and t-tests.
    Plots per-subject comparison.
    """
    # Ensure dataframes are sorted by subject for correct pairing
    df_fmri = df_fmri.sort_values(by="Subject").reset_index(drop=True)
    df_qsm = df_qsm.sort_values(by="Subject").reset_index(drop=True)

    # Z-score all numerical columns (excluding 'Subject')
    for df in [df_fmri, df_qsm]:
        for col in df.columns[1:]:
            df[col] = stats.zscore(df[col])

    corr_results = []
    ttest_results = []
    regions = ['Hand', 'Foot', 'Tongue']
    lvs = ['LV1', 'LV2']

    for lv in lvs:
        for region in regions:
            col_name = f"{region}_{lv}"
            if col_name in df_qsm.columns and col_name in df_fmri.columns:
                qsm_vals = df_qsm[col_name].values
                fmri_vals = df_fmri[col_name].values
                
                # Pearson and Spearman correlations
                r, p_r = pearsonr(qsm_vals, fmri_vals)
                rho, p_rho = spearmanr(qsm_vals, fmri_vals)
                corr_results.append({
                    "Region": region, "LV": lv,
                    "Pearson r": r, "Pearson p": p_r,
                    "Spearman �": rho, "Spearman p": p_rho
                })

                # Paired t-test
                t_stat, p_val = ttest_rel(qsm_vals, fmri_vals)
                ttest_results.append({
                    'Region': region, 'LV': lv,
                    't-statistic': t_stat, 'p-value': p_val
                })
            else:
                print(f"Warning: Column '{col_name}' not found in both QSM and fMRI dataframes for comparison.")

    df_corr = pd.DataFrame(corr_results).round(4)
    df_ttest = pd.DataFrame(ttest_results).round(4)

    print("\n--- QSM vs fMRI LV Regional Weight Correlations ---")
    print(df_corr)
    df_corr.to_csv(os.path.join(results_dir, "QSM_fMRI_LV_regional_correlations.csv"), index=False)

    print("\n--- QSM vs fMRI LV Regional Weight Paired t-tests ---")
    print(df_ttest)
    df_ttest.to_csv(os.path.join(results_dir, "QSM_fMRI_LV_regional_ttests.csv"), index=False)

    # Plotting comparison
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, lv in enumerate(lvs):
        ax = axs[i]
        # Collect handles for unique legend
        qsm_handle = None
        fmri_handle = None
        
        for region in regions:
            col_name = f"{region}_{lv}"
            if col_name in df_qsm.columns and col_name in df_fmri.columns:
                qsm_vals = df_qsm[col_name].values
                fmri_vals = df_fmri[col_name].values

                x_vals = np.arange(len(df_qsm)) # one x per subject

                for j in range(len(x_vals)):
                    ax.plot([x_vals[j] - 0.1, x_vals[j] + 0.1], [qsm_vals[j], fmri_vals[j]], color='gray', alpha=0.5, zorder=1)
                    
                    p1 = ax.scatter(x_vals[j] - 0.1, qsm_vals[j], color='steelblue', zorder=2)
                    p2 = ax.scatter(x_vals[j] + 0.1, fmri_vals[j], color='darkorange', zorder=2)
                    
                    if j == 0 and region == regions[0]: # Assign handle once for the legend
                        qsm_handle = p1
                        fmri_handle = p2

        ax.set_title(f"{lv} Region Weights Comparison")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(df_qsm["Subject"], rotation=45, ha='right')
        ax.set_ylabel("Z-scored Mean Weight")
        if qsm_handle and fmri_handle:
            ax.legend([qsm_handle, fmri_handle], ['QSM', 'fMRI'], title="Modality")

    fig.suptitle("Per-Subject QSM vs fMRI Region Weights (LV1 and LV2)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, "QSM_vs_fMRI_per_subject_comparison_LV1_LV2.png"), dpi=300)
    plt.close()
    print(f"QSM vs fMRI comparison plot saved to {os.path.join(FIGURES_DIR, 'QSM_vs_fMRI_per_subject_comparison_LV1_LV2.png')}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting PLS Regression Analysis ---")

    # 1. Load and prepare data
    df_als_behav_filtered, king_stage_labels = load_behavioral_data(
        ALS_BEHAV_PATH_OLD, ALS_BEHAV_PATH_NEW, ALS_QSM_SUBJECTS
    )
    
    # Filter king_stage_labels to match ALS_QSM_SUBJECTS

    # To ensure correct pairing, we should get labels directly from filtered behavioral df.
    king_stage_labels = df_als_behav_filtered['King Stage'].values
    
    # Select only the ALSFRS columns for Y data
    alsfrs_columns = [col for col in df_als_behav_filtered.columns if 'ALSFRS' in col]
    Y_data_raw = df_als_behav_filtered[alsfrs_columns].values
    Y_data = np.nan_to_num(stats.zscore(Y_data_raw, axis=0)) # Z-score behavioral data

    QSM_data = load_qsm_data(QSM_ALS_DIR, MASK_PATH, ALS_QSM_SUBJECTS)
    X_data = QSM_data.reshape(len(ALS_QSM_SUBJECTS), -1) # Flatten QSM data for PLS
    n_voxels = QSM_data.shape[1] # Number of voxels after masking

    print(f"Loaded {len(ALS_QSM_SUBJECTS)} ALS subjects with QSM data of shape {QSM_data.shape}.")
    print(f"Loaded behavioral data for {len(Y_data)} subjects.")

    # 2. Perform PLS Analysis with LOO
    pls_results = perform_pls_analysis(
        X_data, Y_data, king_stage_labels,
        n_components=2, n_voxels=n_voxels,
        qsm_subjects=ALS_QSM_SUBJECTS, ref_image_path=REF_IMAGE_PATH
    )

    # 3. Plot PLS Scores and MSE
    plot_pls_scores_and_mse(
        pls_results, ALS_QSM_SUBJECTS,
        os.path.join(FIGURES_DIR, 'scores_KingStage_QSM.png')
    )

    # 4. Plot Behavioral Loadings
    plot_behavioral_loadings(
        pls_results['pls_model_final'], alsfrs_columns, # Pass the behavioral columns used
        os.path.join(FIGURES_DIR, 'PLS_R_Behavioral_Loadings_LV1_LV2.png')
    )

    # 5. Save and Plot Subject LV Projections
    lv_df = pd.DataFrame({
        'Subject': ALS_QSM_SUBJECTS,
        'LV1_fMRI': [x[0] for x in pls_results['x_latent_test_all']], # Extract LV1
        'LV2_fMRI': [x[1] for x in pls_results['x_latent_test_all']], # Extract LV2
        'LV1_Behavior': [y[0] for y in pls_results['y_latent_test_all']],
        'LV2_Behavior': [y[1] for y in pls_results['y_latent_test_all']],
        'KingStage': pls_results['test_labels_all']
    })
    lv_df.to_csv(os.path.join(FIGURES_DIR, "subject_LV1_LV2_scores.csv"), index=False)
    print(f"Subject LV scores saved to {os.path.join(FIGURES_DIR, 'subject_LV1_LV2_scores.csv')}")

    plot_subject_lv_projection(
        pls_results, ALS_QSM_SUBJECTS,
        os.path.join(FIGURES_DIR, 'LV1_LV2_subject_projection_QSM.png'),
        king_stage_labels # Pass the actual King Stage labels for coloring
    )

    # 6. Project LV weights to brain space and save NIfTI files
    # The original code here used `pls` which would be the last fitted model from LOO,
    # but `pls_results['pls_model_final']` is more explicit for the full-data model.
    project_lv_to_brain(
        pls_results['pls_model_final'], QSM_data, nib.load(MASK_PATH),
        ALS_QSM_SUBJECTS, RESULTS_DIR
    )

    # 7. Plot Brain Projections on Surface (requires nilearn.plotting)
    # Ensure nilearn is installed and working with its plotting backend
    plot_brain_projections(
        ALS_QSM_SUBJECTS, RESULTS_DIR, MASK_PATH, FIGURES_DIR
    )

    # 8. Calculate and Plot Region-wise Mean Weights
    lv1_proj_paths = [os.path.join(RESULTS_DIR, f'LV1_brain_projection_{sub}.nii.gz') for sub in ALS_QSM_SUBJECTS]
    lv2_proj_paths = [os.path.join(RESULTS_DIR, f'LV2_brain_projection_{sub}.nii.gz') for sub in ALS_QSM_SUBJECTS]

    lv1_regional_means_df = get_regional_lv_means(ALS_QSM_SUBJECTS, lv1_proj_paths, HAND_MASK_PATH, FOOT_MASK_PATH, TONGUE_MASK_PATH, REF_IMAGE_PATH)
    lv2_regional_means_df = get_regional_lv_means(ALS_QSM_SUBJECTS, lv2_proj_paths, HAND_MASK_PATH, FOOT_MASK_PATH, TONGUE_MASK_PATH, REF_IMAGE_PATH)
    
    # Combine regional means into one DataFrame for easier access in correlations
    # Naming convention: Hand_LV1, Foot_LV1, etc.
    combined_regional_means_df = pd.DataFrame({'Subject': ALS_QSM_SUBJECTS})
    for col in lv1_regional_means_df.columns:
        if col != 'Subject':
            combined_regional_means_df[f"{col}_LV1"] = lv1_regional_means_df[col]
            combined_regional_means_df[f"{col}_LV2"] = lv2_regional_means_df[col]
    
    combined_regional_means_df.to_csv(os.path.join(RESULTS_DIR, "per_subject_LV_weights.csv"), index=False)
    print(f"Per-subject LV regional weights saved to {os.path.join(RESULTS_DIR, 'per_subject_LV_weights.csv')}")

    plot_per_subject_region_weights(
        lv1_regional_means_df, lv2_regional_means_df,
        os.path.join(FIGURES_DIR, 'per_subject_LV1_LV2_weights_KS_QSM.png')
    )
    plot_average_region_weights(
        lv1_regional_means_df, lv2_regional_means_df,
        os.path.join(FIGURES_DIR, 'mean_LV1_LV2_region_weights_QSM.png')
    )

    # 9. Analyze Correlations with Behavioral Data
    analyze_correlations_with_behavior(
        df_als_behav_filtered, combined_regional_means_df, ALS_QSM_SUBJECTS, RESULTS_DIR
    )

    # 10. Compare QSM vs fMRI Weights (if fMRI data is available)
    # This part requires df_fmri to be present.
    # The original script assumes 'per_subject_LV_weights_ECM.csv' exists.
    # If this file is generated by another script, ensure it's run first.
    fmri_weights_path = os.path.join(BASE_DIR, "results", "per_subject_LV_weights_ECM.csv")
    if os.path.exists(fmri_weights_path):
        df_fmri_weights = pd.read_csv(fmri_weights_path)
        compare_qsm_fmri_weights(
            combined_regional_means_df, df_fmri_weights, RESULTS_DIR
        )
    else:
        print(f"\nSkipping QSM vs fMRI comparison: fMRI weights file not found at {fmri_weights_path}")

    print("\n--- PLS Regression Analysis Complete ---")
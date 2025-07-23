%-----------------------------------------------------------------------
% MATLAB script for SPM 2nd Level Analysis: Two-Sample T-test
%-----------------------------------------------------------------------
% This script sets up and runs a two-sample t-test comparing two groups
% (e.g., ALS patients vs. Controls) based on specified 1st-level contrast
% images using SPM's batch system.
%-----------------------------------------------------------------------

clear all; clc; close all;

% --- Initialize SPM ---
% Add SPM path if not already added
% spm_path = '/path/to/your/spm12'; % <--- Adjust if needed
% addpath(spm_path);
spm('Defaults', 'fMRI');
spm_jobman('initcfg'); % Initialize batch system

%==========================================================================
%                            USER SETTINGS
%==========================================================================

% --- Output Directory ---
% Define the directory where the 2nd-level analysis results (SPM.mat, contrast images) will be saved.
output_dir = '/home/akalyani/Desktop/GLM_2nd_analysis/ALS_vs_Control_con0002_test'; % Be specific

% --- Group Definitions ---
% List of subject identifiers for Group 1 (ALS)
ALS_subs = {'subjects_names'};
% List of subject identifiers for Group 2 (Controls)
Con_subs = {'subjects_names'}; 

% --- Input Contrast File Information ---
% Base directory containing the 1st-level analysis output for Group 1 (ALS)
ALS_con_dir = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project/GLM_1st_analysis_ALS_2';
% Base directory containing the 1st-level analysis output for Group 2 (Controls)
Con_con_dir = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project/GLM_1st_analysis_control2';

% Filename of the 1st-level contrast image to use for the 2nd-level analysis
% NOTE: Ensure this contrast represents the effect of interest at the 1st level.
%       Removed ',1' - SPM usually handles 3D nifti paths correctly. If you
%       encounter issues, you might need to add it back: 'con_0002.nii,1'
Con_filename = 'con_0002.nii';

% --- Optional Settings ---
% Set to 1 to automatically display results after estimation, 0 to skip
run_results_report = 1;
results_thresh_desc = 'FWE'; % 'FWE', 'FDR', or 'none' (for uncorrected)
results_thresh = 0.05;       % Significance threshold (e.g., 0.05 for FWE/FDR, 0.001 for uncorrected)
results_extent = 3;         % Minimum cluster size (voxels)

%==========================================================================
%                     PREPARE INPUTS & CHECK FILES
%==========================================================================

fprintf('Setting up 2nd level analysis: Two-Sample T-test\n');
fprintf('Output directory: %s\n', output_dir);

% --- Create Output Directory ---
if ~exist(output_dir, 'dir')
    fprintf('Output directory does not exist, creating: %s\n', output_dir);
    mkdir(output_dir);
end

% --- Construct Full Paths to Contrast Images ---
fprintf('Constructing paths for 1st level contrast files (%s)...\n', Con_filename);
% Group 1 (ALS) scans
ALS_scans = cellfun(@(sub) fullfile(ALS_con_dir, sub, Con_filename), ALS_subs, 'UniformOutput', false);
% Group 2 (Control) scans
Con_scans = cellfun(@(sub) fullfile(Con_con_dir, sub, Con_filename), Con_subs, 'UniformOutput', false);

% --- Check if all input contrast files exist ---
fprintf('Checking existence of input contrast files...\n');
all_scans = [ALS_scans(:); Con_scans(:)]; % Combine into one list
missing_files = {};
for i = 1:length(all_scans)
    if ~exist(all_scans{i}, 'file')
        missing_files{end+1} = all_scans{i};
    end
end

if ~isempty(missing_files)
    fprintf(2, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf(2, 'ERROR: The following contrast files are missing:\n');
    for i = 1:length(missing_files)
        fprintf(2, '  - %s\n', missing_files{i});
    end
     fprintf(2, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    error('One or more input contrast files not found. Please check paths and filenames. Aborting.');
else
    fprintf('Verified. All %d input contrast files found (%d ALS, %d Controls).\n', ...
            length(all_scans), length(ALS_scans), length(Con_scans));
end

%==========================================================================
%                        DEFINE SPM BATCH JOBS
%==========================================================================
clear matlabbatch; % Clear any existing batch

% --- Job 1: Factorial Design Specification (Two-Sample T-test) ---
matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir}; % Use curly braces for directory path
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = ALS_scans'; % Group 1 scans (ensure column vector)
matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = Con_scans'; % Group 2 scans (ensure column vector)
matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0; % Independence: Yes (0 = independent samples)
matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1; % Variance: Unequal (1 = assume unequal variances)
matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0; % Grand Mean Scaling: No
matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0; % ANCOVA: No

% Covariates: None specified
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});

% Masking: Implicit mask based on input data, no threshold masking
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1; % Threshold masking: None
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1; % Implicit Mask: Yes (mask based on voxels present in all scans)
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''}; % Explicit Mask: None (empty string)

% Global Calculation / Normalization: Defaults (omit/no scaling)
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1; % Global calculation: Omit
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1; % Global mean scaling: No
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1; % Global normalization: None (set to 1)

% --- Job 2: Model Estimation ---
matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File', ...
                                                      substruct('.', 'val', '{}', {1}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), ...
                                                      substruct('.', 'spmmat')); % Dependency on Job 1 output
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0; % Write Residuals: No
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1; % Estimation Method: Classical (Restricted Maximum Likelihood)

% --- Job 3: Contrast Manager ---
matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', ...
                                                 substruct('.', 'val', '{}', {2}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), ...
                                                 substruct('.', 'spmmat')); % Dependency on Job 2 output
% Contrast 1: ALS > Controls
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'ALS > Controls';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1]; % Weight=1 for Group 1 mean, Weight=-1 for Group 2 mean
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none'; % Replicate over sessions: No
% Contrast 2: Controls > ALS
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Controls > ALS';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1]; % Weight=-1 for Group 1 mean, Weight=1 for Group 2 mean
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.delete = 1; % Delete existing contrasts: Yes (start fresh)

% --- Job 4: (Optional) Results Report ---
if run_results_report
    matlabbatch{4}.spm.stats.results.spmmat(1) = cfg_dep('Contrast Manager: SPM.mat File', ...
                                                         substruct('.', 'val', '{}', {3}, '.', 'val', '{}', {1}, '.', 'val', '{}', {1}), ...
                                                         substruct('.', 'spmmat')); % Dependency on Job 3 output
    matlabbatch{4}.spm.stats.results.conspec.titlestr = ''; % Title: Default
    matlabbatch{4}.spm.stats.results.conspec.contrasts = Inf; % Contrasts: All
    matlabbatch{4}.spm.stats.results.conspec.threshdesc = results_thresh_desc; % Threshold type (FWE, FDR, none)
    matlabbatch{4}.spm.stats.results.conspec.thresh = results_thresh; % Threshold value
    matlabbatch{4}.spm.stats.results.conspec.extent = results_extent; % Extent threshold (voxels)
    matlabbatch{4}.spm.stats.results.conspec.conjunction = 1; % Conjunction: No
    matlabbatch{4}.spm.stats.results.conspec.mask.none = 1; % Masking: None for results display
    matlabbatch{4}.spm.stats.results.units = 1; % Units: Volumetric (1)
    matlabbatch{4}.spm.stats.results.export = {}; % Export: None (can specify e.g., {'png', 'csv'} )
end

%==========================================================================
%                        RUN SPM BATCH JOBS
%==========================================================================

fprintf('\nBatch jobs defined. Starting SPM job manager...\n');
spm_jobman('run', matlabbatch);
fprintf('\nSPM batch processing complete.\n');
fprintf('Results saved in: %s\n', output_dir);

% Return to the original directory if needed (optional)
% cd(original_dir);

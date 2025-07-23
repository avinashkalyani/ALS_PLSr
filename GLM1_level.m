clear all; clc;

%==========================================================================
% USER SETTINGS - Adapted from ALS_project Script & User Input
%==========================================================================

% --- Subject List ---
% Define the subjects to process for this run

names = {"names_of_subjects"};
% --- Paths from ALS_project ---
% Base path for the original ALS project data (timing files, rawish functional)
als_base_path  = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project';
% Directory containing the MNI-registered functional data used in ALS_project
als_dir_func   = fullfile(als_base_path, 'CON_AA_MNI_152T1_1mm_outputs_b3_masked11');
% Directory containing the timing (.ons/.dur) files for ALS_project tasks
als_dir_timing = fullfile(als_base_path, 'control');
% Base directory where the GLM output (SPM.mat, contrast images) will be saved
glm_output_base = fullfile('/home/akalyani/mounts/zdv/Users/akalyani/ALS_project', 'GLM_1st_analysis_control2');

% --- Paths for this script's context (less critical if using preprocessed data) ---
% Base directory containing the subject folders (used for looping)
% Note: Actual functional/timing data paths are now absolute based on ALS_project settings
dir_base        = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_project/control'; % Base for subject *folder* structure if needed, but data comes from als_* paths
script_dir      = '/home/akalyani/mounts/zdv/Users/akalyani/ALS_Project/codes'; % Location of this script

% --- Acquisition & GLM Parameters from ALS_project & User Input ---
n_sess          = 1;        % Number of sessions
n_scans         = 302;      % Scans per session (volumes)
TR              = 2.0;      % Repetition time in seconds
num_slices      = 189;      % Number of slices per volume (metadata)
ref_slice       = 40;       % Reference slice used during *original* slice timing (metadata)
smooth_fwhm     = [2 2 2];  % Smoothing kernel FWHM in mm (applied if smoothed file not found)
cutoff_highpass = 100;      % Cutoff for high-pass filter in seconds
start_analysis  = 3;        % Start directly with GLM estimation (skip preprocessing)
                            % 1=realign/slicetime, 2=norm/smooth, 3=design/estimate, 4=contrasts
rp              = 0;        % Include realignment parameters? 0=no (consistent with ALS_project script)

% --- Conditions from ALS_project ---
cond_names = { ...
    'timeFootLeft','timeFootRight', ...
    'timeHandLeft','timeHandRight', ...
    'timeTongue'};
cond_param = [ 0 0 0 0 0 ]; % Number of parametric modulators per condition (none here)

% --- Contrasts from ALS_project ---
c_name = { ...      % Renamed from contrast_names for consistency with script 2 structure
    'FootLeft_vs_all','FootRight_vs_all', ...
    'HandLeft_vs_all','HandRight_vs_all', ...
    'Tongue_vs_all'};
c_con  = [ ...      % Renamed from c_weights, contains weights including constant column
     4 -1 -1 -1 -1 0; ... % Changed from 5 to 4
    -1  4 -1 -1 -1 0; ... % Changed from 5 to 4
    -1 -1  4 -1 -1 0; ... % Changed from 5 to 4
    -1 -1 -1  4 -1 0; ... % Changed from 5 to 4
    -1 -1 -1 -1  4 0 ];  % Changed from 5 to 4
c_type = {'T','T','T','T','T'}; % All T-contrasts

% --- Other Settings (Defaults from Script 2, may not all be used) ---
delete_files	= 0;      % Delete intermediate preprocessing files? (Preprocessing is skipped anyway)

%===========================================================================
% end of user specified settings
%===========================================================================

%---------------------------------------------------------------------------
% Initialize SPM
%---------------------------------------------------------------------------
spm('Defaults','fMRI');
spm_jobman('initcfg');
global defaults;
defaults.stats.maxmem = 2^35; % Set memory limit if needed
fs = filesep; % platform-specific file separator

% Ensure GLM base output directory exists
if ~exist(glm_output_base,'dir'), mkdir(glm_output_base); end

%---------------------------------------------------------------------------
% Loop through all specified subjects
%---------------------------------------------------------------------------
for k = 1:length(names)

    name_subj_k = names{k}; % Current subject ID

    fprintf(1,'==================================\n');
    fprintf(1,'Starting analysis for subject %s\n', name_subj_k);
    fprintf(1,'==================================\n');

    % Define subject-specific output directory
    dir_out = fullfile(glm_output_base, name_subj_k);
    if ~exist(dir_out,'dir'), mkdir(dir_out); end
    fprintf('Outputs for %s will be saved to: %s\n', name_subj_k, dir_out);

    % Define subject-specific input functional file (from ALS project structure)
    % This is the MNI-registered file BEFORE smoothing in the ALS script
    base_func_file = fullfile(als_dir_func, name_subj_k, 'HFC_reg_f_MNI_f_MNI_2.nii');
    if ~exist(base_func_file, 'file')
        error('Base functional file not found for subject %s: %s', name_subj_k, base_func_file);
    end

    % Define the path for the SMOOTHED functional file (input to GLM)
    % This file resides in the GLM output directory structure
    sfunc = fullfile(dir_out, ['s_' spm_file(base_func_file,'basename') '.nii']);

    % Smooth functional image (only if the smoothed file doesn't already exist)
    % Mimics the logic from the ALS_project script
    if ~exist(sfunc, 'file')
        fprintf('Smoothed file not found. Smoothing %s to %s...\n', base_func_file, sfunc);
        spm_smooth(base_func_file, sfunc, smooth_fwhm);
        fprintf('Smoothing complete.\n');
    else
        fprintf('Skipping smoothing; smoothed file already exists: %s\n', sfunc);
    end

    %===========================================================================
    % Preprocessing Sections (Skipped because start_analysis = 3)
    %===========================================================================
    if start_analysis <= 1
        % Realignment & Unwarp (SKIPPED)
        fprintf('Skipping realignment/unwarp section (start_analysis > 1)\n');
    end % if-block for realign & unwarp section

    if start_analysis <= 2
        % Normalization & Smoothing (SKIPPED - Smoothing handled above, Norm assumed done)
        fprintf('Skipping normalization/smoothing section (start_analysis > 2)\n');
    end % if-block for normalisation & smoothing

    %===========================================================================
    % Define Design Matrix & Estimate Parameters
    %===========================================================================
    if start_analysis <= 3

        fprintf('Starting design matrix definition & parameter estimation...\n');
        cd(dir_out); % Change to the subject's output directory

        % Initialize SPM structure
        SPM = struct();
        SPM.dir            = {dir_out}; % Cell array format

        % Timing parameters
        SPM.nscan          = ones(1, n_sess) * n_scans; % Explicitly set number of scans
        SPM.xY.RT          = TR;                      % Repetition Time
        SPM.xBF.T          = 16;                      % Number of time bins per scan (SPM default)
        SPM.xBF.dt         = SPM.xY.RT / SPM.xBF.T; % Calculate time resolution <-- ADD THIS LINE
        % SPM.xBF.T0: Reference time bin. Default is 8 (middle bin).
        % Kept at default assuming slice timing handled in pre-processed input data,
        % or modelling relative to middle time bin is desired.
        % Original ref_slice was 'ref_slice', but not used here to calculate T0.
        SPM.xBF.T0         = 8;
        SPM.xBF.UNITS      = 'secs';                 % Onsets/durations are in seconds
        SPM.xBF.Volterra   = 1;                      % Order of convolution (SPM default)
        SPM.xBF.name       = 'hrf';                  % Canonical HRF (no derivatives)

        % --- Generate Basis Functions (NOW spm_get_bf receives SPM.xBF.dt) ---
        SPM.xBF            = spm_get_bf(SPM.xBF);

        % Design Options
        SPM.xGX.iGXcalc    = 'None';                 % Global normalization
        SPM.xVi.form	   = 'AR(1)';                % Intrinsic autocorrelations model

        % Session Specific Settings
        if n_sess ~= 1
           warning('Script assumes n_sess = 1 based on ALS script. Adjust if necessary.');
           if length(SPM.nscan) ~= n_sess
               error('Length of SPM.nscan must match n_sess.');
           end
        end

        for sess = 1:n_sess % Loop required even for one session

            % High-pass filter
            SPM.xX.K(sess).HParam = cutoff_highpass;

            % Trial specification: onsets, duration
            for c = 1:length(cond_names)
                % Construct paths to timing files using ALS project structure
                ons_file = fullfile(als_dir_timing, name_subj_k, 'functional', [cond_names{c} '.ons']);
                dur_file = fullfile(als_dir_timing, name_subj_k, 'functional', [cond_names{c} '.dur']);

                if ~exist(ons_file, 'file')
                    error('Onset file not found: %s', ons_file);
                end
                 if ~exist(dur_file, 'file')
                    error('Duration file not found: %s', dur_file);
                end

                SPM.Sess(sess).U(c).name = cond_names(c); % Keep as cell for consistency
                SPM.Sess(sess).U(c).ons  = load(ons_file);
                SPM.Sess(sess).U(c).dur  = load(dur_file);
                SPM.Sess(sess).U(c).P(1).name = 'none'; % No parametric modulators
                SPM.Sess(sess).U(c).P(1).h    = 0;
                SPM.Sess(sess).U(c).P(1).P    = []; % Ensure P field exists even if empty
            end

            % User-specified regressors (Realignment Parameters)
            SPM.Sess(sess).C.C = [];    % No realignment parameters included (rp=0)
            SPM.Sess(sess).C.name = {}; % Empty names
        end

        % Specify functional data file (smoothed file)
        % Ensure the file path includes the scan index ',1' if needed by SPM version/functions
        % Script 1 uses ',1', let's keep it for safety.
        % Check if file has the expected number of scans
        V = spm_vol(sfunc);
        if length(V) ~= n_scans
             warning('NIfTI header for %s indicates %d scans, but n_scans is set to %d. Using value from header.', sfunc, length(V), n_scans);
             SPM.nscan(1) = length(V); % Update SPM.nscan based on header if inconsistent
        end
        SPM.xY.P = sfunc; % Use the file path directly; SPM functions usually handle multi-frame NIfTIs


        % Configure and Estimate
        fprintf('Configuring design matrix...\n');
        SPM = spm_fmri_spm_ui(SPM); % Configure the design matrix without GUI interaction

        fprintf('Estimating GLM parameters...\n');
        SPM = spm_spm(SPM); % Estimate the parameters

        fprintf('GLM estimation complete.\n');

    end % if-block for design matrix definition & parameter estimation

    %===========================================================================
    % Define and Compute Contrasts
    %===========================================================================
    if start_analysis <= 4

        fprintf('Defining and computing contrasts...\n');

        % Switch to analysis directory and load SPM.mat
        cd(dir_out);
        if ~exist('SPM.mat', 'file')
             if start_analysis <= 3 % Only error if estimation should have happened
                 error('SPM.mat not found in %s. Estimation may have failed.', dir_out);
             else % If just running contrasts, need SPM.mat
                 fprintf('Loading existing SPM.mat for contrast definition.\n');
                 load('SPM.mat'); % Load existing SPM structure
             end
        elseif exist('SPM','var') && isfield(SPM,'SPMmat') % If SPM struct from estimation exists
             fprintf('Using SPM structure from estimation step.\n');
        else % If SPM.mat exists but SPM variable doesn't (e.g. script run in parts)
             fprintf('Loading existing SPM.mat for contrast definition.\n');
             load('SPM.mat');
        end

        % Ensure SPM structure is loaded/available
        if ~exist('SPM','var') || ~isfield(SPM,'xX')
            error('SPM structure not available or incomplete for contrast definition in %s.', dir_out);
        end

        % Create/Clear existing contrasts if necessary
        if isfield(SPM,'xCon') && ~isempty(SPM.xCon)
            fprintf('Clearing existing contrasts before defining new ones.\n');
            SPM.xCon = [];
        end

        % Create F-contrast for 'effects of interest' (optional but common)
        try
            % Indices for task regressors (columns B) and constant/user regressors (columns G)
            iX0     = [SPM.xX.iB SPM.xX.iG];
            SPM.xCon = spm_FcUtil('Set','effects of interest','F','iX0', iX0, SPM.xX.xKXs);
        catch ME
             warning('Could not create "effects of interest" F-contrast: %s', ME.message);
             SPM.xCon = []; % Initialize xCon if it failed
        end

        % Add T-contrasts based on c_name, c_con, c_type
        n_cntr_start = length(SPM.xCon); % Number of contrasts defined so far (e.g., F-contrast)

        % Verify contrast vector length matches design matrix columns
        % Expected columns = (num_conditions * num_basis_funcs) + num_regressors + num_sessions_constants
        num_conditions = length(cond_names);
        num_basis_funcs = size(SPM.xBF.bf, 2); % Should be 1 for canonical HRF
        num_regressors = 0; % Since rp=0
        if isfield(SPM.Sess,'C') && isfield(SPM.Sess(1).C,'C')
             num_regressors = size(SPM.Sess(1).C.C, 2); % Get actual number (should be 0)
        end
        num_sessions_constants = n_sess; % One constant per session
        expected_cols = (num_conditions * num_basis_funcs) + num_regressors + num_sessions_constants;

        % Check against columns in actual design matrix X
        actual_cols = size(SPM.xX.X, 2);
        if actual_cols ~= expected_cols
            warning('Mismatch between expected (%d) and actual (%d) design matrix columns. Using actual count.', expected_cols, actual_cols);
            expected_cols = actual_cols; % Use the actual column count for validation
        end

        if size(c_con, 2) ~= expected_cols
             error(['Contrast matrix column count (%d) does not match actual design matrix columns (%d). '...
                    'Check c_con definition and ensure it aligns with conditions (%d * %d b.f.) + regressors (%d) + constants (%d).'], ...
                    size(c_con, 2), expected_cols, num_conditions, num_basis_funcs, num_regressors, num_sessions_constants);
        end

        for c = 1:length(c_name)
            % c_con contains the full contrast vector weights (including constant) from ALS script
            contrast_vector = c_con(c,:);

            % Define the contrast using spm_FcUtil
            % Ensure contrast vector is a column vector for spm_FcUtil
            SPM.xCon(c + n_cntr_start) = spm_FcUtil('Set', c_name{c}, c_type{c}, 'c', contrast_vector', SPM.xX.xKXs);
        end

        % Evaluate contrasts (writes con*.nii, spmT*.nii files)
        fprintf('Evaluating contrasts...\n');
        spm_contrasts(SPM);

        fprintf('Contrast definition and computation complete.\n');

    end     % if-block for contrast definition & evaluation

    % Clear variables for next subject
    clear SPM base_func_file sfunc ons_file dur_file V; % Added V to clear
    fprintf('Finished subject %s.\n\n', name_subj_k);

end     % End of main loop over subjects

fprintf(1,'==================================\n');
fprintf(1,'All selected subjects processed.\n');
fprintf(1,'==================================\n');

cd(script_dir); % Return to the script directory
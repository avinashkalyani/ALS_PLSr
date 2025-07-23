# Individualized Phenotyping of Functional ALS Pathology

This repository contains code and instructions for reproducing the analyses described in the manuscript:

> **Individualized Phenotyping of Functional ALS Pathology in Sensorimotor Cortex**
> *Avinash Kalyani¹ et al.*

---

## Data Availability

Extracted structural data (MP2RAGE, QSM, cortical metrics):
[ALS\_Data.xlsx](https://github.com/alicianorthall/In-vivo-Pathology-ALS/blob/main/ALS_Data.xlsx)

---

## ALS Neuroimaging Analysis Pipeline

A collection of scripts designed to perform a comprehensive neuroimaging analysis pipeline for Amyotrophic Lateral Sclerosis (ALS) patient data, focusing on structural and functional MRI. The pipeline covers:

* Anatomical alignment
* Functional data preprocessing and masking
* Functional connectivity analysis
* GLM-based fMRI analysis (first- and second-level)
* robust Shared Response Model (rSRM) analysis
* Partial Least Squares Regression (PLSR) correlating neuroimaging features with behavioral scores

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Dependencies](#dependencies)
5. [Data Preparation](#data-preparation)
6. [Usage Workflow](#usage-workflow)
7. [Results](#results)
8. [Contributing](#contributing)
9. [Citation](#citation)

---

## Introduction

This project analyzes neuroimaging data from ALS patients and healthy controls to identify brain–behavior relationships, focusing on how brain activity and connectivity relate to ALS progression and symptom manifestation (e.g., King Stage and ALSFRS‑R scores). The workflow leverages FSL, ANTs, AFNI, SPM12, and Python libraries such as NiBabel, Nilearn, Scikit‑learn, and Brainiak.

---

## Project Structure

```
ALS_Project/
├── als/
│   ├── als_list.txt
│   └── <subject_id>/
│       ├── anatomical/<subject_id>_T1_WB.nii.gz
│       └── functional/
│           ├── adata.nii.gz
│           └── time<BodyPart>.ons
├── control/
│   ├── control_list.txt
│   └── <subject_id>/
│       ├── anatomical/<subject_id>_T1_WB.nii.gz
│       └── functional/
│           ├── adata.nii.gz
│           └── time<BodyPart>.ons
├── behavioral_data/
│   ├── ALS_behavioral.csv
│   ├── New_ALS_behave.csv
│   └── control_behavioral.csv
├── mni_icbm152_t1_tal_nlin_sym_09a_masked.nii
├── CON_AA_MNI_152T1_1mm_outputs_b3_masked/
│   └── <subject_id>/...
│        ├── HFC_reg_f_MNI_f_MNI_2.nii.gz
├── ALS_AA_MNI_152T1_1mm_outputs_b3_masked/
│    └── <subject_id>/...
│        ├── HFC_reg_f_MNI_f_MNI_2.nii.gz
├── QSM_MNI_outputs_re_registered/...
├── combined_functional_localizer_masks/
├── CON_functional_localizer_masks/
├── ALS_functional_localizer_masks/
├── GLM_1st_analysis_control2/
├── GLM_1st_analysis_ALS_2/
├── GLM_2nd_analysis/
├── results/
├── figures/
└── scripts/
    ├── Anatomical_alignment.py
    ├── NL_AA_mni.py
    ├── cleaned_threshold_statmaps.py
    ├── connected_cluster_masks_tasks.py
    ├── ECM_HFC.sh
    ├── GLM1_level.m
    ├── Group_2nd_2sample_test.m
    ├── ALS_vs_control_rSRM.py
    ├── PLSR_BOLD.py
    └── PLSR_ECM.py
```

> **Note:** Update hardcoded paths in scripts to match your local data organization.

---

## Features

* **Anatomical Alignment**: Linear and non-linear T1 alignment to MNI using ANTs.
* **Functional Preprocessing & Masking**: Thresholding and morphological refinement of fMRI statistical maps (FSL, AFNI).
* **Functional Connectivity (ECM)**: High-Frequency Connectivity mapping via vecm and vnifti.
* **GLM Analysis (SPM12)**: First- and second-level models for task-based fMRI.
* **Shared Response Model (rSRM)**: Aligns multi-subject fMRI data and classifies ALS subgroups.
* **PLSR**: Relates BOLD and ECM features to behavioral scores with cross-validation.

---

## Dependencies

### Neuroimaging Software

* **FSL**: fslstats, fslmaths
* **ANTs**: antsRegistrationSyNQuick.sh, antsRegistration
* **AFNI**: 3dresample, 3dmaskdump
* **SPM12** in MATLAB
* **LIPSIA** or similar toolbox for vecm, vnifti

### Python Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nibabel nilearn scipy brainiak
```

---

## Data Preparation

1. Organize raw and preprocessed data per [Project Structure](#project-structure).
2. Prepare T1 and BOLD fMRI data (slice-time corrected, realigned, normalized).
3. Generate timing files (`.ons` and `.dur`) for task events.
4. Place behavioral CSVs in `behavioral_data/` with subject IDs, ALSFRS‑R, King Stage.
5. Ensure MNI template is available:

   * `mni_icbm152_t1_tal_nlin_sym_09a_masked.nii`

---

## Usage Workflow

1. **Anatomical Alignment**:

   * `python scripts/Anatomical_alignment.py` (linear)
   * `python scripts/NL_AA_mni.py` (non-linear)

2. **Functional Mask Creation**:

   * `python scripts/cleaned_threshold_statmaps.py`
   * `python scripts/connected_cluster_masks_tasks.py`

3. **Timeseries Extraction**:

   * AFNI commands in scripts for 3dresample & 3dmaskdump

4. **Functional Connectivity (ECM)**:

   * Run `bash scripts/ECM_HFC.sh`

5. **GLM Analysis (SPM)**:

   * In MATLAB, open and run `GLM1_level.m` for first level
   * Run `Group_2nd_2sample_test.m` for second level

6. **Shared Response Model**:

   * `python scripts/ALS_vs_control_rSRM.py`

7. **PLSR**:

   * `python scripts/PLSR_BOLD.py`
   * `python scripts/PLSR_ECM.py`

---

## Results

Outputs generated under `results/` and `figures/` include:

* **NIfTI**: Aligned T1s, binary masks, PLS weight maps
* **NPZ**: Preprocessed timeseries data
* **CSV**: Behavioral scores, LV correlations
* **Plots**: Confusion matrices, permutation histograms, PLS score/loadings
* **SPM**: `SPM.mat`, `con_*.nii`, `spmT_*.nii`

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug reports and feature requests.

---

## Citation

If you use this code or data, please cite:

Kalyani, A., et al. (202x). *Individualized Phenotyping of Functional ALS Pathology in Sensorimotor Cortex*. NeuroImage. DOI\:xxx

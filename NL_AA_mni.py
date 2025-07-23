import os
import concurrent.futures
import multiprocessing

# Path to the MNI template
mni_template = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii"

# Main working directory
main_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/als"

# Create output directory
output_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/ALS_AA_MNI_152T1_1mm_outputs_b3_masked2"
os.makedirs(output_dir, exist_ok=True)

# Function to perform registration
def perform_registration(subject_dir):
    subject_id = os.path.basename(subject_dir)
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    subject_t1 = os.path.join(subject_dir,'anatomical', f"%s_T1_WB.nii.gz" % subject_id)
    # Check if the subject output directory exists
    if os.path.exists(subject_output_dir):
        # Check if the subject directory contains any .nii.gz files
        nii_files = [file for file in os.listdir(subject_output_dir) if file.endswith(".nii.gz")]
        if not nii_files:
            print(f"No .nii.gz files found in {subject_id}. Performing registration.")
            
            # Perform registration using ANTs
            os.system(f"antsRegistration \
                --verbose 1 \
                --dimensionality 3 \
                --float 1 \
                --output [{os.path.join(subject_output_dir, 'registered_')},{os.path.join(subject_output_dir, 'registered_Warped.nii.gz')},{os.path.join(subject_output_dir, 'registered_InverseWarped.nii.gz')}] \
                --interpolation Linear \
                --use-histogram-matching 0 \
                --winsorize-image-intensities [0.005,0.995] \
                --transform Affine[0.1] \
                --metric MI[{mni_template},{subject_t1},0.7,32,Regular,0.1] \
                --convergence [1000x500,1e-6,10] \
                --shrink-factors 2x1 \
                --smoothing-sigmas 1x0vox \
                --transform SyN[0.1,2,0] \
                --metric CC[{mni_template},{subject_t1},1,2] \
                --convergence [500x100,1e-6,10] \
                --shrink-factors 2x1 \
                --smoothing-sigmas 1x0vox")
            
            print(f"Alignment completed for {subject_id}")
        else:
            print(f".nii.gz files found in {subject_id}. Skipping registration.")
    else:
        print(f"Output directory does not exist for {subject_id}. Skipping registration.")

if __name__ == "__main__":
    # Get a list of subject directories
    subject_dirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    # Process 4 subjects at a time using multiprocessing.Pool
    with multiprocessing.Pool(processes=6) as pool:
        pool.map(perform_registration, subject_dirs)

    print("Alignment completed for all subjects!")

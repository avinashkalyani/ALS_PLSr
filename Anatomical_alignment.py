import os
import concurrent.futures
import multiprocessing

#/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/als/bo58/anatomical/bo58_T1_WB.nii.gz


# Path to the MNI template
mni_template = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/mni_icbm152_t1_tal_nlin_sym_09a_masked.nii"

# Main working directory
main_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/als"

# Create output directory
output_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/ALS_AA_MNI_152T1_1mm_outputs_b3_masked"
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
        

            # Perform registration using ANTs (SyN)
            os.system(f"antsRegistrationSyNQuick.sh -d 3 -t b -f {mni_template} -m {subject_t1} -o {os.path.join(subject_output_dir, f'T1_aligned_')}")

            print(f"Alignment completed for {subject_id}")
        else:
            print(f".nii.gz files found in {subject_id}. Skipping registration.")
    else:
        print(f"Output directory does not exist for {subject_id}. Skipping registration.")


if __name__ == "__main__":
    # Get a list of subject directories
    subject_dirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    # Process 4 subjects at a time using multiprocessing.Pool
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(perform_registration, subject_dirs)


    print("Alignment completed for all subjects!")

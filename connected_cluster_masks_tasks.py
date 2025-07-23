import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_erosion


def find_largest_cluster(binary_mask):
    """
    Finds the largest connected cluster in a 3D binary mask.
    """
    structure = generate_binary_structure(3, 1)  # 6-connectivity
    labeled_mask, num_labels = label(binary_mask, structure)
    if num_labels == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    label_counts = np.bincount(labeled_mask.flatten())
    largest_label = np.argmax(label_counts[1:]) + 1  # skip background
    return labeled_mask == largest_label


def apply_morphology(mask, dilate_iters=2, erode_iters=2):
    """
    Applies binary dilation followed by erosion to a binary mask.
    """
    mask_dilated = binary_dilation(mask, iterations=dilate_iters)
    return binary_erosion(mask_dilated, iterations=erode_iters)


def load_and_combine_masks(subject_dirs, body_parts):
    """
    Loads and combines binary masks for given body parts across subjects.
    """
    combined_masks = {}
    for part in body_parts:
        combined = None
        for subject_dir in subject_dirs:
            mask_path = os.path.join(subject_dir, f"VOXEL_MASK_bin_1501_{part}.nii.gz")
            try:
                mask = nib.load(mask_path).get_fdata() > 0
            except FileNotFoundError:
                print(f"File not found: {mask_path}")
                continue

            combined = mask if combined is None else combined + mask

        if combined is not None:
            combined_masks[part] = combined
    return combined_masks


def save_nifti_mask(mask, reference_path, output_path):
    """
    Saves a binary mask as a NIfTI file using the affine of a reference image.
    """
    reference_image = nib.load(reference_path)
    nifti_img = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_image.affine)
    nib.save(nifti_img, output_path)


if __name__ == "__main__":
    # Define paths and body parts
    base_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project"
    subject_base = os.path.join(base_dir, "CON_AA_MNI_152T1_1mm_outputs_b3_masked11")
    output_dir = os.path.join(base_dir, "CON_functional_localizer_masks")
    reference_img_path = os.path.join(base_dir, "mni_icbm152_t1_tal_nlin_sym_09a_masked.nii")
    body_parts = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue', 'tongue_hemi']

    os.makedirs(output_dir, exist_ok=True)

    # Collect subject directories
    subject_dirs = [
        os.path.join(subject_base, d) for d in os.listdir(subject_base)
        if os.path.isdir(os.path.join(subject_base, d))
    ]

    # Combine and process masks
    combined_masks = load_and_combine_masks(subject_dirs, body_parts)

    for part, combined_mask in combined_masks.items():
        print(f"Processing: {part}")
        largest_cluster = find_largest_cluster(combined_mask > 0)
        processed_mask = apply_morphology(largest_cluster, dilate_iters=2, erode_iters=2)

        output_path = os.path.join(output_dir, f"largest_cluster_mask_{part}_processed.nii.gz")
        save_nifti_mask(processed_mask, reference_img_path, output_path)
        print(f"Saved: {output_path}")

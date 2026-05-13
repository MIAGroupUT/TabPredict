import os
import re

def rename_umcg_image_files(target_dir):
    """
    Renames UMCG NIfTI files in place by removing the last digit of the ID.
    Example: 3841_UMCG.nii.gz -> 384_UMCG.nii.gz
    """
    
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist.")
        return

    # Regex to capture the 4-digit ID and the site
    # ^(\d{3}) -> Group 1: First 3 digits
    # (\d)     -> Group 2: The 4th digit (the '1' we want to remove)
    # _UMCG\.nii\.gz$ -> Matches the suffix
    pattern = re.compile(r"^(\d{3})(\d)_UMCG\.nii\.gz$")

    files_processed = 0

    print(f"Scanning directory: {target_dir}\n")

    # List files first to avoid issues with directory state changing during iteration
    for filename in os.listdir(target_dir):
        match = pattern.match(filename)
        
        if match:
            prefix = match.group(1)   # e.g., '384'
            # match.group(2) is the '1'
            
            new_name = f"{prefix}_UMCG.nii.gz"
            
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_name)
            
            # Safety: Don't overwrite if 384_UMCG.nii.gz already exists
            if os.path.exists(new_path):
                print(f"Skipping: {filename} -> Target {new_name} already exists.")
                continue

            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                files_processed += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    print(f"\nTask complete. Total UMCG images renamed: {files_processed}")

if __name__ == "__main__":
    # Update this to your imagesTr directory
    IMAGES_DIR = "/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/imagesTr_pre-op"

    rename_umcg_image_files(IMAGES_DIR)

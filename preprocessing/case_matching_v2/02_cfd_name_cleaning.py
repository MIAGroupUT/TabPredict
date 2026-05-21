import os
import shutil
import re

def copy_and_rename_radar_files(source_dir, target_dir):
    """
    Copies files following the RADAR_CTA_{ID}_... pattern to a target directory
    and renames them to {ID}_MST.vtp.
    """
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    # Regex to capture the ID (the digits after RADAR_CTA_)
    # Pattern: RADAR_CTA_ followed by digits, then underscores/text
    pattern = re.compile(r"RADAR_CTA_(\d+)_.*\.vtp")

    files_processed = 0

    for filename in os.listdir(source_dir):
        match = pattern.match(filename)
        
        if match:
            # Extract the ID (e.g., 118)
            file_id = match.group(1)
            new_name = f"{file_id}_MST.vtp"
            
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, new_name)
            
            try:
                shutil.copy2(source_path, target_path)
                print(f"Copied: {filename} -> {new_name}")
                files_processed += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

    print(f"\nTask complete. Total files processed: {files_processed}")

if __name__ == "__main__":
    # Update these paths to your specific local directories
    SOURCE_PATH = "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/ensemble_predictions_v3--AAA-full-all"
    TARGET_PATH = "/deepstore/datasets/mia/AI4EVAR/raw_data/CFDs/renamed"

    copy_and_rename_radar_files(SOURCE_PATH, TARGET_PATH)
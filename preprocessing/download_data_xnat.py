import os
import xnat

# Configuration
XNAT_HOST = 'https://research-xnat.utwente.nl'
USER_NAME = 'ALIAS_TOKEN'  # <-- Paste alias token from xnat here (use token secret as password)
BASE_OUTPUT_DIR = '/deepstore/datasets/mia/AI4EVAR/raw_data/RADAR_XNat/data' 

PROJECT_IDS = [
    'ARTinEVAR-AUMC',
    'ARTinEVAR-UMCG',
    'ARTinEVAR',    # RNST
    'AllARTinEVARRijnstatePat', # Rijnstate NEW
    'ARTinEVAR-MST',
]

print(f"Connecting to {XNAT_HOST}...")

with xnat.connect(XNAT_HOST, user=USER_NAME) as session:
    print("Connection successful!")
    
    for project_id in PROJECT_IDS:
        print(f"\n" + "="*40)
        print(f"Starting download for project: {project_id}")
        print("="*40)
        
        # Define the target directory path
        project_output_dir = os.path.join(BASE_OUTPUT_DIR, f"{project_id}_data")
        
        # AUTOMATIC MKDIR: Creates the folders if they don't exist yet
        if not os.path.exists(project_output_dir):
            print(f"Creating directory: {project_output_dir}")
            os.makedirs(project_output_dir, exist_ok=True)
        
        try:
            project = session.projects[project_id]
            
            # Start the download into the newly created folder
            project.download_dir(project_output_dir)
            print(f"Success! {project_id} downloaded to: {project_output_dir}")
            
        except KeyError:
            print(f"ERROR: Project '{project_id}' not found or access denied.")
        except Exception as e:
            print(f"ERROR downloading '{project_id}': {e}")

print("\nAll scheduled downloads completed!")

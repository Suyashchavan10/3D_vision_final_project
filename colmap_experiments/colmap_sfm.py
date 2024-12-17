# import os
# import subprocess

# def run_colmap(image_path, output_path):
#     colmap_executable = r"C:\Users\suyash\IIITB\colmap-x64-windows-nocuda\COLMAP.bat"  # Assuming 'colmap' is in the system PATH, or provide full path to colmap.exe

#     # Create database path
#     database_path = os.path.join(output_path, 'database.db')

#     # 1. Feature Extraction
#     print("Running feature extractor...")
#     feature_extractor_cmd = [
#         colmap_executable, 'feature_extractor',
#         '--database_path', database_path,
#         '--image_path', image_path
#     ]
#     subprocess.run(feature_extractor_cmd, check=True)
    
#     # 2. Feature Matching (Exhaustive Matcher)
#     print("Running exhaustive matcher...")
#     exhaustive_matcher_cmd = [
#         colmap_executable, 'exhaustive_matcher',
#         '--database_path', database_path
#     ]
#     subprocess.run(exhaustive_matcher_cmd, check=True)
    
#     # 3. 3D Reconstruction (Mapper)
#     print("Running mapper...")
#     mapper_cmd = [
#         colmap_executable, 'mapper',
#         '--database_path', database_path,
#         '--image_path', image_path,
#         '--output_path', output_path
#     ]
#     subprocess.run(mapper_cmd, check=True)
    
#     print("COLMAP process completed successfully.")

# # Usage example
# image_directory = r"C:\Users\suyash\IIITB\3D_vision_final_project\images\bottle"  # image directory
# output_directory = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle"  # output directory

# # Create the output directory if it doesn't exist
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# run_colmap(image_directory, output_directory)

import os
import subprocess

def run_colmap(image_path, output_path):
    colmap_executable = r"C:\Users\suyash\IIITB\colmap-x64-windows-nocuda\COLMAP.bat"  # Assuming 'colmap' is in the system PATH, or provide full path to colmap.exe

    # Create database path
    database_path = os.path.join(output_path, 'database.db')

    # 1. Feature Extraction
    print("Running feature extractor...")
    feature_extractor_cmd = [
        colmap_executable, 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_path
    ]
    subprocess.run(feature_extractor_cmd, check=True)
    
    # 2. Feature Matching (Exhaustive Matcher)
    print("Running exhaustive matcher...")
    exhaustive_matcher_cmd = [
        colmap_executable, 'exhaustive_matcher',
        '--database_path', database_path
    ]
    subprocess.run(exhaustive_matcher_cmd, check=True)
    
    # 3. 3D Reconstruction (Mapper)
    print("Running mapper...")
    mapper_cmd = [
        colmap_executable, 'mapper',
        '--database_path', database_path,
        '--image_path', image_path,
        '--output_path', output_path
    ]
    subprocess.run(mapper_cmd, check=True)

    # 4. Export the model to text files (instead of binary files)
    print("Exporting model to .txt files...")
    export_cmd = [
        colmap_executable, 'model_converter',
        '--input_path', os.path.join(output_path, '0'),  # Assuming the output model is in folder '0'
        '--output_path', os.path.join(output_path, 'txt_model'),  # Folder to save the .txt files
        '--output_type', 'TXT'
    ]
    subprocess.run(export_cmd, check=True)

    print("COLMAP process completed successfully. Model exported as .txt files.")

# Usage example
image_directory = r"C:\Users\suyash\IIITB\3D_vision_final_project\images\bottle"  # image directory
output_directory = r"C:\Users\suyash\IIITB\3D_vision_final_project\colmap_experiments\colmap_output\bottle"  # output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

run_colmap(image_directory, output_directory)


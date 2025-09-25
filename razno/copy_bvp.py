import os
import shutil

# Define the source and destination base folders
source_base = '/Volumes/externiMacNovi/UBFC-Phys/yolo/no_segmentation/dataset'  # Change this
destination_base = 'bvp_info_values'  # Change this

# Create the destination base folder if it doesn't exist
# os.makedirs(destination_base, exist_ok=True)

# Loop through s1 to s56
for i in range(1, 57):
    folder_name = f's{i}'
    source_folder = os.path.join(source_base, folder_name)
    destination_folder = os.path.join(destination_base, folder_name)

    # Create the new folder
    os.makedirs(destination_folder, exist_ok=True)

    info_filename = f'info_{folder_name}.txt'
    source_info_file = os.path.join(source_folder, info_filename)
    destination_info_file = os.path.join(destination_folder, info_filename)
    # Copy the file if it exists
    if os.path.exists(source_info_file):
        shutil.copy2(source_info_file, destination_info_file)
    else:
        print(f'File not found: {source_info_file}')

    # Define the filenames to copy
    for t in range(1, 4):
        filename = f'bvp_{folder_name}_T{t}.csv'
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)

        # Copy the file if it exists
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
        else:
            print(f'File not found: {source_file}')

import os
# provjeri postoje li svi csv fajlovi u svim subject folderima
# Path to the root directory containing s1, s2, ..., s56
base_path = "/Volumes/stari/UBFC-Phys/faceBoxes/segment_30s/framerate_1/rppg"
# base_path = "/Volumes/externiMacNovi/UBFC-Phys/faceBoxes/segment_10s/framerate_1/rppg"
num_of_segments = 6

# Expected task and segment configuration
tasks = ["T1", "T2", "T3"]
segments = [f"seg{str(i).zfill(2)}" for i in range(num_of_segments)]

missing_files = []

for subject_num in range(1, 57):  # s1 to s56
    folder_name = f"s{subject_num}"
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.isdir(folder_path):
        print(f"Folder missing: {folder_name}")
        continue

    for task in tasks:
        for seg in segments:
            expected_filename = f"rppg_s{subject_num}_{task}_{seg}.csv"
            file_path = os.path.join(folder_path, expected_filename)
            if not os.path.isfile(file_path):
                missing_files.append(file_path)

# Report results
if missing_files:
    print(f"\nMissing files ({len(missing_files)} total):")
    for file in missing_files:
        print(file)
else:
    print("All expected files are present.")
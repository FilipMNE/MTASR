import os
import subprocess

base_path = "/Volumes/externiMacNovi/UBFC-Phys/no_segmentation/dataset"  # originalni dataset sa videima od 3 minuta

# ovo dvoje mijenjas u zavisnosti od toga kolika je duzina segmenta
new_dataset_path = "/Volumes/stari/UBFC-Phys/segment_30s/dataset"  # Adjust this path if needed
segment_len = 30  # seconds

for i in range(45, 57):  # From s1 to s56
    folder_name = f"s{i}"
    person_path = os.path.join(base_path, folder_name)
    
    if not os.path.isdir(person_path):
        print(f"Skipping missing folder: {person_path}")
        continue

    # Create inner folder with the same name
    new_subject_path = os.path.join(new_dataset_path, folder_name)
    if os.path.exists(new_subject_path):
        continue
    os.makedirs(new_subject_path, exist_ok=True)

    for t in range(1, 4):  # T1, T2, T3
        video_filename = f"vid_{folder_name}_T{t}.avi"
        input_path = os.path.join(person_path, video_filename) # putanja za originalni video
        output_pattern = os.path.join(new_subject_path, f"vid_{folder_name}_T{t}_seg%02d.avi")
        
        if not os.path.isfile(input_path):
            print(f"Video file not found: {input_path}")
            continue

        # izgled komande:
        # ffmpeg -i vid_s1_T2.avi -c copy -map 0 -segment_time 10 -f segment -reset_timestamps 1 ../../dataset_shorter/s1/vid_s1_T2_seg%02d.avi
        # Construct ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", str(segment_len),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]

        print(f"Processing {input_path}")
        subprocess.run(cmd, check=True)

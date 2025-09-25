import cv2
import pandas as pd
import matplotlib.pyplot as plt 

video_path = '/Users/filipjovanovic/Desktop/Multimedija/UBFC-Phys/s1/vid_s1_T1.avi'
def get_frames_from_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    frame_num=0
    while success:
        if frame_num % 500 == 0: #vraca svaki 500. frejm
            count += 1
            cv2.imwrite("frames/s1/T1/frame_%d_.jpg" % count, image)     # save frame as JPEG file
            print("Created frame %d" % count)   
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        frame_num += 1

if __name__ == '__main__':
    # get_frames_from_video(video_path)

    file_path = "/Volumes/externiMAC/UBFC-Phys/dataset/s1/rppg/rppg_all_s1_T1.csv"
    df = pd.read_csv(file_path)

    for column in df.columns:
        print(df[column])
        plt.figure()
        plt.title(column)
        plt.plot(df[column])
        plt.show()
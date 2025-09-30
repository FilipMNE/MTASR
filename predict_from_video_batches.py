import cv2
import os
import sys
import numpy as np
import pandas as pd
import math
from PIL import Image
import yaml
from utils.POS import Pulse
from utils.filter import butter_bandpass_filter, detrend
import torch
import io
import time

from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image
from pos_rppg import skin_segment, zero_mean
from data_preprocess import z_score, find_peak

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))
# drop_num = cfg['train']['drop_num']
# step = cfg['train']['step']
# clip_len = cfg['train']['clip_len']
drop_num = 35
step = 60
clip_len = 120

# video_path = "/Volumes/externiMAC/UBFC-Phys/dataset/s1/vid_s1_T3.avi" # test - ima stresa
# video_path = "/Users/filipjovanovic/Desktop/Multimedija/UBFC-Phys/s2/vid_s2_T3.avi"
# video_path = '/Users/filipjovanovic/Desktop/OneAI/VISION/V4V/V4V dataset/Phase 1_ Training_Validation sets/Videos/train-001_of_002/F001_T1.mkv'
video_path = '/Users/filipjovanovic/Downloads/1000012036.mp4'
# video_path = "/Volumes/externiMAC/UBFC-Phys/dataset/s2/vid_s2_T2.avi" # control - nema stresa
dataset_root = "./tmp_values"


def get_rppg(X):
    pulse = Pulse(35, len(X))
    bvp = pulse.get_pulse(X)
    bvp = zero_mean(bvp)
    bvp = detrend(bvp)
    bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5)
    save_path = dataset_root
    print("SAVE PATH for rppg:", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if math.isnan(bvp[0]):
        print("nan")
    np.savetxt(os.path.join(save_path, r'rppg_batches.csv'), bvp, fmt='%f',
               delimiter=',')
    print('Kreirao rppg_batches.csv')

def get_data(save_datas):
    person_path = dataset_root # za rppg.csv

    ppg_signal = pd.read_csv(os.path.join(person_path, r"rppg_batches.csv"), header=None)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)
    ppg_signal = ppg_signal[drop_num:]

    total_times = (len(ppg_signal) - clip_len) / step + 1
    if total_times < 1:
        total_times = 1

    for times in range(math.floor(total_times)):
        ppg_start_index = int(times * step)
        ppg_end_index = int(ppg_start_index + clip_len)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]
        vpg_signal_ = np.gradient(ppg_signal_)
        
        ppg_signal_ = z_score(ppg_signal_)
        vpg_signal_ = z_score(vpg_signal_)

        save_datas.append(ppg_signal_)

# # Function to run inference
# def run_inference(bvp_signal, net_type="both"):
#     """
#     Runs inference on a single BVP signal.

#     :param bvp_signal: NumPy array or PyTorch tensor of shape (1, 2100)
#     :param net_type: "hr" for heart rate estimation, "both" for classification & HR
#     :return: Model output (classification, HR, peaks depending on net_type)
#     """
#     # Convert input to tensor if necessary
#     if isinstance(bvp_signal, np.ndarray):
#         bvp_signal = torch.tensor(bvp_signal, dtype=torch.float32)

#     # Ensure correct shape (batch_size=1, channels=1, sequence_length=2100)
#     bvp_signal = bvp_signal.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 2100)

#     # Run inference
#     with torch.no_grad():
#         if net_type == "hr":
#             hr, p_peak = model(bvp_signal, net_type)
#             return hr.cpu().numpy(), p_peak.cpu().numpy()
#         elif net_type == "both":
#             outputs, hr, p_peak = model(bvp_signal, net_type)
#             predicted_class = torch.argmax(outputs, dim=1).item()
#             return predicted_class, hr.cpu().numpy(), p_peak.cpu().numpy()


if __name__ == '__main__':
    arg_num = len(sys.argv)
    if arg_num > 1: # keira novi rppg ako se u command line pozove sa jos jednim argumentom, inace odma ide na preprocesiranje
#****************************************** pos_rppg ********************************************************************************************
        vidcap = cv2.VideoCapture(video_path)

        count = 0 #  koliko frejmova ce biti obradjeno u face_detection -> npr. svaki 10. ili 35.
        frame_num = 0 # koliko ukupno frejmova je procitano -> uvijek oko 6325
        frame_rate = 1 # koliki je step u citanju frejmova

        encode_format = 'png' # ili 'jpg'
        batch = np.zeros((7000, 128, 128, 3))

        batch_of_frames = []
        batch_of_frames_size = 600

        start_time = time.time() # mjeri vrijeme izvrsavanja
        while True:

            # Napuni batch_of_frames sa 600 frejmova (batch_of_frames_size)
            print("Reading frames from video...")
            for _ in range(batch_of_frames_size):
                success, image = vidcap.read()
                if not success:
                    break
                frame_num += 1
                if frame_num % frame_rate == 0:
                    batch_of_frames.append((image, frame_num))

            # Ako je batch prazan, izadji iz while petlje -> predji na sljedeci video
            if not batch_of_frames:
                break

            # Konvertuj procitane frejmove u BytesIO format
            batch_of_frames_bytes = []
            for frame, frame_num_from_dict in batch_of_frames:
                # encode_format promijeni u .png -> bolji kvalitet ali malo sporije, sacuvaj jpg .csv dok uporedis rezultat
                success, encoded_frame = cv2.imencode(".{0}".format(encode_format), frame)
                if success:
                    frame_bytes = encoded_frame.tobytes()
                    contents = io.BytesIO(frame_bytes)
                    batch_of_frames_bytes.append((contents, frame_num_from_dict))
                else:
                    raise ValueError("Could not encode frame to image format.")
            batch_of_frames.clear()

            # YOLO i obrada contents bajtova
            for contents, frame_num in batch_of_frames_bytes:
                # YOLO - detekcija lica
                predicted_array = eval_image(contents)

                if isinstance(predicted_array, tuple): # ako nije nadjeno lice na frejmu, preskoci ga
                    print("Nije nadjeno lice na frejmu", frame_num)
                    contents.seek(0) # Make sure we're at the start of the stream
                    face_not_found_image = Image.open(contents)
                    face_not_found_image.save(os.path.join(dataset_root, "face_not_found/frejm_{0}.jpg".format(frame_num)))
                    continue

                # ako je pronadjeno lice na frejmu
                if frame_num % 35 == 0: # stampaj svaki 35. da vidis da radi program
                    print("Found face on frame {0}".format(frame_num))

                face_image = Image.fromarray(predicted_array.astype(np.uint8))
                frame_face = np.array(face_image)

                # If needed, convert RGB to BGR for OpenCV compatibility
                if frame_face.ndim == 3 and frame_face.shape[2] == 3:
                    frame_face = cv2.cvtColor(frame_face, cv2.COLOR_RGB2BGR)
                
                frame_face = cv2.resize(frame_face, (128, 128))
                frame_face = skin_segment(frame_face)  
                batch[count] = frame_face
                count += 1

            #-----------------------------------
        print(f"Time taken to get rppg from video using batches: {time.time() - start_time:.2f} seconds")
        
        print("Racuna rppg...")
        batch = batch[:count]

        mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
        b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
        mean_rgb = np.stack((r, g, b), axis=1)
        get_rppg(mean_rgb)

    # data_preprocess ********************************************************************************************************************************
    print("Preprocessing csv data")
    save_datas = []
    get_data(save_datas)

    # predict ********************************************************************************************************************************
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_rad_trening_modeli/trained_models/faceBoxes/segment_10s_framerate_1_pp1_hr0.75/MTASR_pa0.75_hr1_2_120.pth", map_location=device, weights_only=False)
    model.to(device)
    model.eval()  # Set to evaluation mode

    input_data = np.stack(save_datas, axis=0) # shape: (4,2100)
    input_data = torch.tensor(input_data, dtype=torch.float32)
    input_data = input_data.unsqueeze(1) # shape should be: Tensor([4,1,2100])

    net_type = "both" # ili "hr"
    if net_type == "hr":
            hr, p_peak = model(input_data, net_type)
            print(hr, p_peak)
    elif net_type == "both":
        outputs, hr, p_peak = model(input_data, net_type)
        predicted_class = torch.argmax(outputs, dim=1)
        print(predicted_class, hr, p_peak)
        print("OUTPUTS:", outputs)
    print("Predicted successfully")

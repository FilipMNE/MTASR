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

from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image
from pos_rppg import skin_segment, zero_mean
from data_preprocess import z_score, find_peak

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))
drop_num = cfg['train']['drop_num']
step = cfg['train']['step']
clip_len = cfg['train']['clip_len']

video_path = "/Volumes/externiMAC/UBFC-Phys/dataset/s1/vid_s1_T3.avi" # test - ima stresa
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
    np.savetxt(os.path.join(save_path, r'rppg.csv'), bvp, fmt='%f',
               delimiter=',')
    print('Kreirao rppg.csv')


def get_data(save_datas):
    person_path = dataset_root # za rppg.csv

    ppg_signal = pd.read_csv(os.path.join(person_path, r"rppg.csv"), header=None)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)
    ppg_signal = ppg_signal[drop_num:]

    total_times = (len(ppg_signal) - clip_len) / step + 1

    for times in range(math.floor(total_times)):
        # print(times, ' put usao u petlju')
        # ppg_start_index = img_index + times_index - (self.clip_len - 1) * self.jump_num
        ppg_start_index = int(times * step)
        ppg_end_index = int(ppg_start_index + clip_len)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]
        vpg_signal_ = np.gradient(ppg_signal_)

        # ppg_signal_ = outlier(ppg_signal_)
        # ppg_signal_ = butter_bandpass_filter(ppg_signal_, 35)
        # ppg_signal_ = detrend(ppg_signal_)
        # ppg_signal_ = norm(ppg_signal_)
        ppg_signal_ = z_score(ppg_signal_)
        vpg_signal_ = z_score(vpg_signal_)

        peak_ = find_peak(ppg_signal_)
        vpg_peak_ = find_peak(vpg_signal_)
        
        save_datas.append(ppg_signal_)



# Function to run inference
def run_inference(bvp_signal, net_type="both"):
    """
    Runs inference on a single BVP signal.

    :param bvp_signal: NumPy array or PyTorch tensor of shape (1, 2100)
    :param net_type: "hr" for heart rate estimation, "both" for classification & HR
    :return: Model output (classification, HR, peaks depending on net_type)
    """
    # Convert input to tensor if necessary
    if isinstance(bvp_signal, np.ndarray):
        bvp_signal = torch.tensor(bvp_signal, dtype=torch.float32)

    # Ensure correct shape (batch_size=1, channels=1, sequence_length=2100)
    bvp_signal = bvp_signal.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 2100)

    # Run inference
    with torch.no_grad():
        if net_type == "hr":
            hr, p_peak = model(bvp_signal, net_type)
            return hr.cpu().numpy(), p_peak.cpu().numpy()
        elif net_type == "both":
            outputs, hr, p_peak = model(bvp_signal, net_type)
            predicted_class = torch.argmax(outputs, dim=1).item()
            return predicted_class, hr.cpu().numpy(), p_peak.cpu().numpy()


if __name__ == '__main__':
    arg_num = len(sys.argv)
    if arg_num > 1: # keira novi rppg ako se u command line pozove sa jos jednim argumentom, inace odma ide na preprocesiranje
        # pos_rppg ********************************************************************************************************************************
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0 #  koliko frejmova ce biti obradjeno u face_detection
        frame_num = 0 # koliko ukupno frejmova je procitano
        frame_rate = 1 # koliki je step u citanju frejmova
        while success:
            if frame_num % frame_rate == 0: #vraca svaki 35. frejm (jer frame_rate == 35)
                count += 1  
                cv2.imwrite(os.path.join(dataset_root, "tmp.jpg"), image)    # save frame as temporary JPG file, svi se cuvaju u jednoj jpg 
                contents = read_from_path(os.path.join(dataset_root, "tmp.jpg"))

                # obrada frejma - detekcija lica
                predicted_array = eval_image(contents)

                if isinstance(predicted_array, tuple): # ako nije nadjeno lice na frejmu, preskoci ga
                    print("Nije nadjeno lice na frejmu", count)
                    cv2.imwrite(os.path.join(dataset_root, "face_not_found/frejm_{0}.jpg".format(count)), image)
                else:
                    face_image = Image.fromarray(predicted_array.astype(np.uint8))
                    face_image.save(os.path.join(dataset_root, "tmp_rppg/{0}.jpg".format(count)))
                    print("Predicted face in image frame_{0}_.jpg".format(count))

            success,image = vidcap.read()
            frame_num += 1

        # u tmp_rppg se nalaze fokusirane slike lica, sad nad tim folderom treba da se pozovu funkcije iz pos_rppg.py
        print("Racuna rppg")
        faces_path = os.path.join(dataset_root, "tmp_rppg") # folder gdje se nalaze slike detektovanih faca
        faces_files_names = os.listdir(faces_path) # imena slika: 1.jpg, 2.jpg, 3.jpg,...

        # jer mi kreira .DS_Store file
        for fname in faces_files_names:
            if fname.startswith("."):
                faces_files_names.remove(fname)

        # sortira da bi vidio koliko ih ima ukupno
        faces_files_names.sort(key=lambda x: int(x.split('.')[0]))
        total_length = int(faces_files_names[-1].split('.')[0])
        # print("broj slika:", total_length)

        batch = np.zeros((total_length, 128, 128, 3))
        for index in range(total_length):
            frame_face = np.array(cv2.imread(os.path.join(faces_path, "{0}.jpg".format(index+1))))
            frame_face = cv2.resize(frame_face, (128, 128))
            frame_face = skin_segment(frame_face)
            batch[index] = frame_face
        mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
        b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
        mean_rgb = np.stack((r, g, b), axis=1)
        get_rppg(mean_rgb)

    # data_preprocess ********************************************************************************************************************************
    print("Preprocessing data")
    save_datas = []
    get_data(save_datas)


    # za uporedjivanje sa stvarnim podacima ----------------------------------------------------------------
    # data = np.load(r"ubfc_phys_new.npy", allow_pickle=True)
    # real_val = data[8:12, 4]

    # diff = 0
    # for my_data, real_data in zip(save_datas, real_val):
    #     for x, y in zip(my_data, real_data):
    #         diff += abs(x-y)
    # print("Error:", diff)
    # ------------------------------------------------------------------------------------------------------

    # predict ********************************************************************************************************************************
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("trained_models/MTASR_pa0.75_hr1_5_2100.pth", map_location=device, weights_only=False)
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
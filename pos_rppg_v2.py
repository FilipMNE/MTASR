import os
import io
import yaml
import cv2
import numpy as np
from utils.filter import butter_bandpass_filter, detrend
import math
import time

from io import BytesIO
from PIL import Image, ImageOps
from ultralytics import YOLO
from PIL import ImageDraw

from utils.POS import Pulse
from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image

yoloModel = YOLO("face_detection/FaceDetectionYolo/yolov11n-face.pt")

data_root = r"/Volumes/externiMAC/UBFC-Phys/dataset"
save_root = r"/Volumes/externiMAC/UBFC-Phys"

yaml_file = "setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))


def get_rppg(X, dir_, task, part):
    pulse = Pulse(35, len(X))
    bvp = pulse.get_pulse(X)
    bvp = zero_mean(bvp)
    bvp = detrend(bvp)
    bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5)
    save_path = os.path.join(save_root, r'rppg/{0}'.format(dir_))
    print("Save path for rppg:", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if math.isnan(bvp[0]):
        print("nan")
    np.savetxt(os.path.join(save_path, r'rppg_{0}_{1}_T{2}.csv'.format(part, dir_, task)), bvp, fmt='%f',
               delimiter=',')
    print('Kreirao rppg_{0}_{1}_T{2}.csv'.format(part, dir_, task))


def skin_segment(bgr_image):
    ycrcb_image = None
    try:
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    except Exception as e:
        print(e)
        print(bgr_image.shape)
    H, W, C = ycrcb_image.shape
    mask = np.zeros((H, W), dtype="uint8")
    cb = ycrcb_image[:, :, 2]
    cr = ycrcb_image[:, :, 1]
    y = ycrcb_image[:, :, 0]
    cb_index = np.logical_and(cb > 77, cb < 127)
    cr_index = np.logical_and(cr > 137, cr < 177)
    y_index = np.logical_and(y > 80, y < 255)
    mask[np.logical_and(cb_index, cr_index, y_index)] = 1
    SKIN_ROI = cv2.add(bgr_image, np.zeros(np.shape(bgr_image), dtype=np.uint8), mask=mask)
    return SKIN_ROI


def zero_mean(signal):
    return signal - np.mean(signal)


# za s14 T1: 726.25 sekundi
if __name__ == '__main__':

    dataset_root = "/Volumes/externiMAC/UBFC-Phys"
    dataset = "/Volumes/externiMAC/UBFC-Phys/dataset"
    havent_done = ['s14', 's15', 's16', 's17']
    # ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']

    for person in havent_done:
        person_path = os.path.join(dataset, person)

        for task in [1,2,3]:
            start_time = time.time()
            start_time_500 = time.time()
            video_path = os.path.join(person_path, "vid_{0}_T{1}.avi".format(person, task))
            vidcap = cv2.VideoCapture(video_path)
            
            success,image = vidcap.read()
            count = 0 #  koliko frejmova ce biti obradjeno u face_detection (npr. svaki 10. ili 35. ili svaki)
            frame_num = 0 # koliko frejmova je procitano -> uvijek oko 6325
            frame_rate = 1 # koliki je step u citanju frejmova

            batch = np.zeros((7000, 128, 128, 3))

            while success:
                frame_num += 1
                if frame_num % frame_rate == 0: #vraca svaki {frame_rate}. frejm
                    # Konvertuje procitani frejm u BytesIO
                    success, encoded_frame = cv2.imencode('.jpg', image)
                    if success:
                        frame_bytes = encoded_frame.tobytes()
                        contents = io.BytesIO(frame_bytes)
                    else:
                        raise ValueError("Could not encode frame to image format.")

                    # YOLO - obrada frejma - detekcija lica
                    predicted_array = eval_image(contents)

                    if isinstance(predicted_array, tuple): # ako nije nadjeno lice na frejmu, preskoci ga
                        print("Nije nadjeno lice na frejmu", frame_num)
                        cv2.imwrite(os.path.join(dataset_root, "face_not_found/{0}T{1}frejm_{2}.jpg".format(person, task, frame_num)), image)
                        success,image = vidcap.read()
                        continue

                    # ako je pronadjeno lice na frejmu
                    count += 1
                    if frame_num % 100 == 0:
                        print("Found face for subject {0}, task {1}, frame {2}".format(person, task, frame_num))
                    face_image = Image.fromarray(predicted_array.astype(np.uint8))
                    frame_face = np.array(face_image)

                    # If needed, convert RGB to BGR for OpenCV compatibility
                    if frame_face.ndim == 3 and frame_face.shape[2] == 3:
                        frame_face = cv2.cvtColor(frame_face, cv2.COLOR_RGB2BGR)
                    
                    frame_face = cv2.resize(frame_face, (128, 128))
                    frame_face = skin_segment(frame_face)  
                    batch[count-1] = frame_face

                    if frame_num % 500 == 0:
                        print(f"Time taken to read 500 frames: {time.time() - start_time_500:.2f} seconds")
                        start_time_500 = time.time()


                success,image = vidcap.read()
            
            print(f"Time taken to read frames: {time.time() - start_time:.2f} seconds")
            print("Racuna rppg za", person, "task", task)
            batch = batch[:count]

            mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
            b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
            mean_rgb = np.stack((r, g, b), axis=1)
            get_rppg(mean_rgb, person, task, "all")

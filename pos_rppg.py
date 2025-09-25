import os
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

# data_root = r"/home/som/8T/DataSets/ubfc_phys/3_part"
# save_root = r"/home/som/8T/DataSets/ubfc_phys/pos_rppg"
data_root = r"/Volumes/externiMAC/UBFC-Phys/dataset"
save_root = r"/Volumes/externiMAC/UBFC-Phys/dataset"

yaml_file = "setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))


def get_rppg(X, dir_, task, part):
    pulse = Pulse(35, len(X))
    bvp = pulse.get_pulse(X)
    bvp = zero_mean(bvp)
    bvp = detrend(bvp)
    bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5)
    save_path = os.path.join(save_root, r'{0}/rppg'.format(dir_))
    print("SAVE PATH for rppg:", save_path)
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


if __name__ == '__main__':

    dataset_root = "/Volumes/externiMAC/UBFC-Phys"
    dataset = "/Volumes/externiMAC/UBFC-Phys/dataset"
    havent_done = ['s1', 's2', 's3', 's4', 's5']
    # ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']

    for person in havent_done:
        person_path = os.path.join(dataset, person)

        for task in [1,2,3]:
            video_path = os.path.join(person_path, "vid_{0}_T{1}.avi".format(person, task))
            vidcap = cv2.VideoCapture(video_path)
            success,image = vidcap.read()
            count = 0 #  koliko frejmova ce biti obradjeno u face_detection
            frame_num=0 # koliko frejmova je procitano
            frame_rate = 1 # koliki je step u citanju frejmova
            while success:
                if frame_num % frame_rate == 0: #vraca svaki 35. frejm (jer frame_rate == 35)
                    count += 1  
                    cv2.imwrite(os.path.join(dataset_root, "tmp/tmp.jpg"), image)    # save frame as temporary JPG file, svi se cuvaju u jednoj jpg 
                    contents = read_from_path(os.path.join(dataset_root, "tmp/tmp.jpg"))

                    # obrada frejma - detekcija lica
                    predicted_array = eval_image(contents)

                    if isinstance(predicted_array, tuple): # ako nije nadjeno lice na frejmu, preskoci ga
                        print("Nije nadjeno lice na frejmu", count)
                        cv2.imwrite(os.path.join(dataset_root, "face_not_found/{0}T{1}frejm_{2}.jpg".format(person, task, count)), image)
                    else:
                        face_image = Image.fromarray(predicted_array.astype(np.uint8))
                        face_image.save(os.path.join(dataset_root, "tmp_rppg/{0}.jpg".format(count)))
                        print("Predicted face in image {0}/T{1}/frame_{2}_.jpg".format(person, task, count))

                success,image = vidcap.read()
                frame_num += 1
            
            # u tmp_rppg se nalaze fokusirane slike lica, sad nad tim folderom treba da se pozovu funkcije iz pos_rppg.py
            print("Racuna rppg za", person, "task", task)
            faces_path = os.path.join(dataset_root, "tmp_rppg") # folder gdje se nalaze slike detektovanih faca
            faces_files_names = os.listdir(faces_path) # imena slika: 1.jpg, 2.jpg, 3.jpg,...

            # jer mi kreira .DS_Store file
            for fname in faces_files_names:
                if fname.startswith("."):
                    faces_files_names.remove(fname)

            # sortira da bi vidio koliko ih ima ukupno
            faces_files_names.sort(key=lambda x: int(x.split('.')[0]))
            total_length = int(faces_files_names[-1].split('.')[0])
            print("broj slika:", total_length)

            batch = np.zeros((total_length, 128, 128, 3))
            for index in range(total_length):
                frame_face = np.array(cv2.imread(os.path.join(faces_path, "{0}.jpg".format(index+1))))
                frame_face = cv2.resize(frame_face, (128, 128))
                frame_face = skin_segment(frame_face)
                batch[index] = frame_face
            mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
            b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
            mean_rgb = np.stack((r, g, b), axis=1)
            get_rppg(mean_rgb, person, task, "all")


    #------------------------------------------------------------------------------------------------
    # STARI KOD - ne radi

    # havent_done = ['s1', 's2', 's3', 's4', 's5']
    # ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    # havent_done = ['s48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    # havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47']

    # for dir_ in havent_done:
    #     person_path = os.path.join(data_root, dir_)
    #     for task in [1, 2, 3]:
    #         print(person_path, task)

    #         face_path = os.path.join(person_path, r"T{0}/face".format(task))
    #         files_name_face = os.listdir(face_path)

    #         # jer mi kreira .DS_Store file
    #         for fname in files_name_face:
    #             if fname.startswith("."):
    #                 files_name_face.remove(fname)

    #         files_name_face.sort(key=lambda x: int(x.split('.')[0]))

    #         total_lengh = int(files_name_face[-1].split('.')[0])

    #         batch = np.zeros((total_lengh, 128, 128, 3))
    #         for index in range(total_lengh):
    #             frame_face = np.array(cv2.imread(os.path.join(face_path, "{0}.jpg".format(index+1))))
    #             frame_face = cv2.resize(frame_face, (128, 128))
    #             frame_face = skin_segment(frame_face)
    #             batch[index] = frame_face
    #         mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)
    #         b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
    #         mean_rgb = np.stack((r, g, b), axis=1)
    #         get_rppg(mean_rgb, dir_, task, "all")

    #------------------------------------------------------------------------------------------------
    #ZA BRISANJE

    # import matplotlib.pyplot as plt
    #
    # face_path = "E:\\dataset\\ubfc-phys\\3_part\\s11\\T1\\face"
    # files_name_face = os.listdir(face_path)
    # files_name_face.sort(key=lambda x: int(x.split('.')[0]))
    #
    # total_lengh = int(files_name_face[-1].split('.')[0])
    #
    # mean_bgr = []
    # for index in range(total_lengh):
    #     frame_face = np.array(cv2.imread(os.path.join(face_path, "{0}.jpg".format(index))))
    #     # frame_face = cv2.resize(frame_face, (128, 128))
    #     frame_face = skin_segment(frame_face)
    #     frame_mean_bgr = frame_face.sum(axis=(0, 1)) / (frame_face != 0).sum(axis=(0, 1))
    #     mean_bgr.append(frame_mean_bgr)
    # # mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)))
    # mean_bgr = np.array(mean_bgr)
    # b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
    # # b, g, r = zero_mean(b), zero_mean(g), zero_mean(r)
    # # b, g, r = detrend(b), detrend(g), detrend(r)
    # # b, g, r = butter_bandpass_filter(b, 35), butter_bandpass_filter(g, 35), butter_bandpass_filter(r, 35)
    # mean_rgb = np.stack((r, g, b), axis=1)
    #
    # pulse = Pulse(35, len(mean_rgb))
    # bvp = pulse.get_pulse(mean_rgb)
    # bvp = zero_mean(bvp)
    # bvp = detrend(bvp)
    # bvp = butter_bandpass_filter(bvp, 35)
    # plt.figure()
    # bvp = bvp.tolist()
    # plt.plot(range(len(bvp)), bvp, color='b', linestyle='-')
    # plt.show()
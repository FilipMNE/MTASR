import os
import io
import yaml
import cv2
import numpy as np
import math
import time


from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from utils.POS import Pulse
from utils.filter import butter_bandpass_filter, detrend
from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image

yoloModel = YOLO("face_detection/FaceDetectionYolo/yolov11n-face.pt")

data_root = r"/Volumes/externiMAC/UBFC-Phys/dataset"
save_root = r"/Volumes/externiMAC/UBFC-Phys"

yaml_file = "setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))


def get_rppg(X, dir_, task, part):
    pulse = Pulse(35, len(X)) # 35 -> video framerate
    bvp = pulse.get_pulse(X) # objasnjena u utils/POS.py
    bvp = zero_mean(bvp) # Removes the mean value from the signal to center it around zero.
    bvp = detrend(bvp) # Removes slow-varying trends (like lighting changes or motion) to isolate the true heartbeat frequency component.
    bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5) # Applies a Butterworth bandpass filter between 0.7 and 2.5 Hz (≈ 42–150 bpm) to extract the clean heart rate signal.
    if math.isnan(bvp[0]): # Simple check to see if the signal failed (e.g., due to division by zero or numerical instability).
        print("nan")

    # cuvanje rppg signala u .csv fajl
    save_path = os.path.join(save_root, r'rppg/{0}'.format(dir_))
    print("Save path for rppg:", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savetxt(os.path.join(save_path, r'rppg_{0}_{1}_T{2}_AAA.csv'.format(part, dir_, task)), bvp, fmt='%f',
               delimiter=',')
    print('Kreirao rppg_{0}_{1}_T{2}_AAA.csv'.format(part, dir_, task))


def skin_segment(bgr_image):
    '''This function performs skin segmentation using YCrCb color space thresholds
     — it identifies which pixels likely correspond to skin, then creates a masked image with only those pixels visible.
     Y: brightness (luminance), Cr/ Cb: chrominance (color components).
    '''

    #Converts the input image from OpenCV’s default BGR to YCrCb color space.
    ycrcb_image = None
    try:
        ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
    except Exception as e:
        print(e)
        print(bgr_image.shape)
    H, W, C = ycrcb_image.shape

    # Extracts the individual Y, Cr, and Cb channels.
    cb = ycrcb_image[:, :, 2]
    cr = ycrcb_image[:, :, 1]
    y = ycrcb_image[:, :, 0]

    # Create boolean masks based on thresholds
    # These define the skin color range in the YCrCb color space.
    # Only pixels satisfying all three conditions are considered skin.
    cb_index = np.logical_and(cb > 77, cb < 127)
    cr_index = np.logical_and(cr > 137, cr < 177)
    y_index = np.logical_and(y > 80, y < 255)
    mask = np.zeros((H, W), dtype="uint8")
    # Sets mask[y, x] = 1 where pixel is likely skin.
    mask[np.logical_and(cb_index, cr_index, y_index)] = 1

    # Apply the mask to the original image
    SKIN_ROI = cv2.add(bgr_image, np.zeros(np.shape(bgr_image), dtype=np.uint8), mask=mask)
    # A new image (SKIN_ROI) of the same shape (128, 128, 3) where:
    # Only skin-colored regions are visible.
    # Non-skin areas are blacked out.
    return SKIN_ROI


def zero_mean(signal):
    return signal - np.mean(signal)


if __name__ == '__main__':

    dataset_root = "/Volumes/externiMAC/UBFC-Phys"
    dataset = "/Volumes/externiMAC/UBFC-Phys/dataset"
    havent_done = ['s56']
    # ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
    #                's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
    #                's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
    #                's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']

    for person in havent_done:
        person_path = os.path.join(dataset, person)

        for task in [1,2,3]:
            # TODO provjeri s31 T3, mozda postoji vise face_nout_found frejmova nego sto je nasao
            # if person == 's55' and task < 2:
            #     continue
            video_path = os.path.join(person_path, "vid_{0}_T{1}.avi".format(person, task))
            vidcap = cv2.VideoCapture(video_path)

            count = 0 #  koliko frejmova ce biti obradjeno u face_detection (npr. svaki 10. ili 35. ili svaki)
            frame_num = 0 # koliko frejmova je procitano -> uvijek oko 6325
            frame_rate = 1 # koliki je step u citanju frejmova

            encode_format = 'png' # ili 'jpg'
            batch = np.zeros((7000, 128, 128, 3))

            batch_of_frames = []
            batch_of_frames_size = 500 # velicina za buffer za citanje frejmova; izmedju 100 i 1000, obicno najbrze radi na 600
            start_time = time.time() # mjeri vrijeme izvrsavanja
            while True:
                # Napuni batch_of_frames sa 500 frejmova (batch_of_frames_size) - treba oko 6 sekundi
                print('Reading a new batch of frames...')
                reading_from_disk_start_time = time.time()
                for _ in range(batch_of_frames_size):
                    success, image = vidcap.read()
                    if not success:
                        break
                    frame_num += 1
                    if frame_num % frame_rate == 0:
                        batch_of_frames.append((image, frame_num))
                # print(f"Time taken to read {len(batch_of_frames)} frames from disk: {time.time() - reading_from_disk_start_time:.2f} seconds")
                
                # Ako je batch prazan, izadji iz while petlje -> predji na sljedeci video
                if not batch_of_frames:
                    break

                # Konvertuj procitane frejmove u BytesIO format
                # ovaj dio koda je sporiji od citanja frejmova sa diska
                batch_of_frames_bytes = []
                print("Converting frames to BytesIO format...")
                convert_to_bytes_start_time = time.time()
                for frame, frame_num_from_pair in batch_of_frames:
                    # encode_format promijenjen iz .jpg u .png -> bolji kvalitet ali malo sporije, razliciti rezultati .csv fajla za .jpg i .png
                    # if frame_num_from_pair % 100 == 0:
                    #     imencode_start_time = time.time()
                    success, encoded_frame = cv2.imencode(".{0}".format(encode_format), frame) # ovaj linija je prespora -> 0.0215 sekundi po frejmu, * 500 frejmova = 11 sekundi
                    # if frame_num_from_pair % 100 == 0:
                    #     print(f"Time taken to encode frame {frame_num_from_pair} to {encode_format}: {time.time() - imencode_start_time:.6f} seconds")
                    if success: # ovaj dio koda se brzo izvrsava
                        frame_bytes = encoded_frame.tobytes()
                        contents = io.BytesIO(frame_bytes)
                        batch_of_frames_bytes.append((contents, frame_num_from_pair))
                    else:
                        raise ValueError("Could not encode frame to image format.")
                # print(f"Time taken to convert {len(batch_of_frames_bytes)} frames to BytesIO: {time.time() - convert_to_bytes_start_time:.2f} seconds")
                batch_of_frames.clear()
                # memorija se oslobadja - smanji opterecenje RAM-a smanjivanjem velicine batch_of_frames ??

                # YOLO i obrada contents bajtova
                process_bytes_start_time = time.time()
                for contents, frame_num in batch_of_frames_bytes:
                    # if frame_num % 100 == 0:
                    #     yolo_start_time = time.time()
                    # YOLO - detekcija lica
                    predicted_array = eval_image(contents) # ova linija se sporo izvrsava -> 0.079 sekundi po frejmu, * 500 frejmova = 39.5 sekundi
                    # if frame_num % 100 == 0:
                    #     print(f"Time taken to process frame {frame_num} with YOLO: {time.time() - yolo_start_time:.6f} seconds")

                    # ako nije nadjeno lice na frejmu, frejm ne ulazi u rppg, ali ga sacuvaj na disku zbog provjere
                    if isinstance(predicted_array, tuple):
                        print("Nije nadjeno lice na frejmu", frame_num)
                        contents.seek(0) # Make sure we're at the start of the stream
                        face_not_found_image = Image.open(contents)
                        face_not_found_image.save(os.path.join(dataset_root, "face_not_found/{0}T{1}frejm_{2}.jpg".format(person, task, frame_num)))
                        continue

                    # ako je pronadjeno lice na frejmu
                    if frame_num % 175 == 0: # stampaj svaki 175. da vidis da radi program (175frames / 35fps = 5 sekundi videa)
                        print("Found face for subject {0}, task {1}, frame {2}".format(person, task, frame_num))

                    # umjesto ove dvije linije, dovoljno je samo da cast-ujem predicted_array(float32) u .astype(np.uint8) -> isti rezultat
                    # face_image = Image.fromarray(predicted_array.astype(np.uint8))
                    # frame_face = np.array(face_image)
                    frame_face = predicted_array.astype(np.uint8)

                    # If needed, convert RGB to BGR for OpenCV compatibility
                    if frame_face.ndim == 3 and frame_face.shape[2] == 3:
                        frame_face = cv2.cvtColor(frame_face, cv2.COLOR_RGB2BGR)
                    
                    frame_face = cv2.resize(frame_face, (128, 128))
                    frame_face = skin_segment(frame_face)

                    # # za cuvanje izlaza iz skin_segment() funcije, za prezentaciju samo
                    # skin_segment_face = Image.fromarray(frame_face.astype(np.uint8))
                    # skin_segment_face.save("tmp_values/{0}.jpg".format(count))
                    # frame_face.shape() --> (128, 128, 3)

                    batch[count] = frame_face
                    count += 1
                # print(f"Time taken to process {len(batch_of_frames_bytes)} frames: {time.time() - process_bytes_start_time:.2f} seconds")
                
            print(f"Time taken to get rppg from video: {time.time() - start_time:.2f} seconds")

            print("Racuna rppg za", person, "task", task)
            batch = batch[:count]

            #********* objasnjenje ove linije: mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6) *********
            # 1.
            # batch.sum(axis=(1, 2)) --> tells NumPy to sum over the height and width dimensions (128 × 128 pixels).
            # This leaves: 6300 images, For each image, it returns one sum per channel (B, G, R). --> output shape: (6300, 3)

            # 2.
            # batch != 0 --> Produces a boolean array of the same shape (6300, 128, 128, 3) (True where pixel value is !=0)
            # .sum(axis=(1, 2)) --> You get, for each image, the count of non-zero pixels per channel. --> Result shape: (6300, 3)
            # + 1e-6 --> Adds a tiny value (epsilon) to prevent division by zero if any channel has only zero pixels

            # np.true_divide() --> obicni division element-wise
            mean_bgr = np.true_divide(batch.sum(axis=(1, 2)), (batch != 0).sum(axis=(1, 2)) + 1e-6)

            # mijenja redosljed boja iz BGR u RGB
            b, g, r = mean_bgr[:, 0], mean_bgr[:, 1], mean_bgr[:, 2]
            mean_rgb = np.stack((r, g, b), axis=1)
            get_rppg(mean_rgb, person, task, encode_format)

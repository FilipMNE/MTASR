import os
import io
import yaml
import cv2
import numpy as np
import math
import time

from PIL import Image, ImageOps, ImageDraw
from utils.POS import Pulse
from utils.filter import butter_bandpass_filter, detrend
# from face_detection.FaceDetectionYolo.face_detection import read_from_path, eval_image
from face_detection.FaceBoxes.FaceBoxes import FaceBoxes, cropped_img

yaml_file = "setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))

# podesi root_dir, model za face detection, duzinu segmenta i framerate
root_dir = "/Volumes/stari/UBFC-Phys" # zavisi koristis li eksterni disk (stari ili novi) ili lokalni disk, "/Volumes/externiMacNovi/UBFC-Phys" ili "/Volumes/stari/UBFC-Phys" ili "/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_rad_trening"
face_detection_model = 'faceBoxes' # "yolo" ili "faceBoxes"
segment_id = 'segment_30s' # "no_segmentation" ili "segment_10s" ili "segment_30s"
framerate_id = 'framerate_1' # "framerate_1" ili "framerate_5"

dataset_dir = f"{root_dir}/yolo/{segment_id}/dataset" # tu se nalaze folderi 's1', 's2',... - uvijek u yolo folderu
rppg_dir = f"{root_dir}/{face_detection_model}/{segment_id}/{framerate_id}/rppg" # gdje ce se kreirati rppg folder
if not os.path.exists(rppg_dir):
    os.makedirs(rppg_dir)
face_not_found_folder = f"{root_dir}/{face_detection_model}/{segment_id}/face_not_found" # gdje se cuvaju frejmovi na kojima nije pronadjeno lice
if not os.path.exists(face_not_found_folder):
    os.makedirs(face_not_found_folder)

segment_len = cfg[segment_id]['segment_len']
number_of_segments = math.ceil(180 / segment_len)
frame_rate = cfg[segment_id][framerate_id]['frame_rate']
initial_batch_size = cfg[segment_id]['max_num_of_frames']

face_boxes = FaceBoxes()

def get_rppg(X, dir_, task, segment):
    pulse = Pulse(35, len(X)) # 35 -> video framerate
    bvp = pulse.get_pulse(X) # objasnjena u utils/POS.py
    bvp = zero_mean(bvp) # Removes the mean value from the signal to center it around zero.
    bvp = detrend(bvp) # Removes slow-varying trends (like lighting changes or motion) to isolate the true heartbeat frequency component.
    bvp = butter_bandpass_filter(bvp, 35, 0.7, 2.5) # Applies a Butterworth bandpass filter between 0.7 and 2.5 Hz (≈ 42–150 bpm) to extract the clean heart rate signal.
    if math.isnan(bvp[0]): # Simple check to see if the signal failed (e.g., due to division by zero or numerical instability).
        print("nan")

    # cuvanje rppg signala u .csv fajl
    save_path = os.path.join(rppg_dir, f'{dir_}')
    print("Save path for rppg:", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if segment_id == 'no_segmentation':
        filename = r'rppg_png_{0}_T{1}.csv'.format(dir_, task)
        file_path = os.path.join(save_path, filename)
    else:
        filename = r'rppg_{0}_T{1}_seg{2}.csv'.format(dir_, task, segment)
        file_path = os.path.join(save_path, filename)
    np.savetxt(file_path, bvp, fmt='%f', delimiter=',')
    print(f'Kreirao {filename}')

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
    havent_done = ['s40']
    # ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
    #     's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 
    #     's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 
    #     's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 
    #     's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48', 's49', 's50', 
    #     's51', 's52', 's53', 's54', 's55', 's56']

    for person in havent_done:
        person_path = os.path.join(dataset_dir, person)
        for task in [1,2,3]:
            if person == 's40' and task > 1:
                continue
            for segment in range(number_of_segments):
                # if person == 's39' and task == 1 and segment < 15:
                #     continue
                # pretpostavljam da nece biti preko 99 segmenata
                if segment < 10:
                    segment = "0" + str(segment)
                else:
                    segment = str(segment)

                if segment_id == 'no_segmentation':
                    video_path = os.path.join(person_path, "vid_{0}_T{1}.avi".format(person, task))
                else:
                    video_path = os.path.join(person_path, "vid_{0}_T{1}_seg{2}.avi".format(person, task, segment))
                vidcap = cv2.VideoCapture(video_path)

                count = 0 #  koliko frejmova ce biti obradjeno u face_detection -> za koliko frejmova je nasao lice
                frames_read = 0 # koliko frejmova je procitano -> uvijek 6325

                encode_format = 'png' # ili 'jpg', ali png ima bolji kvalitet, ali je sporiji
                
                batch = np.zeros((initial_batch_size, 128, 128, 3))

                batch_of_frames = []
                batch_of_frames_size = 500 # velicina za buffer za citanje frejmova; izmedju 100 i 1000, obicno najbrze radi na 600
                start_time = time.time() # mjeri vrijeme izvrsavanja
                while True:
                    batch_of_frames.clear()
                    # Napuni batch_of_frames sa 500 (batch_of_frames_size) frejmova - treba oko 6 sekundi - fiksirano vrijeme
                    # print('Reading a new batch of frames...')
                    reading_from_disk_start_time = time.time()
                    for _ in range(batch_of_frames_size):
                        success, image = vidcap.read()
                        if not success:
                            break
                        frames_read += 1
                        if frames_read % frame_rate == 0:
                            batch_of_frames.append((image, frames_read))
                    # print(f"Time taken to read {len(batch_of_frames)} frames from disk: {time.time() - reading_from_disk_start_time:.2f} seconds")
                    # Ako je batch prazan, izadji iz while petlje -> predji na sljedeci video
                    if not batch_of_frames:
                        break

                    # Konvertuj procitane frejmove u BytesIO format - ne moramo jer faceBoxes model ne radi sa BytesIO podacima nego cv2 frejmom

                    # FaceBoxes detekcija lica
                    process_bytes_start_time = time.time()
                    for frame, frame_num in batch_of_frames:
                        # if frame_num % 100 == 0:
                        #     fb_start_time = time.time()
                        # FaceBoxes - detekcija lica
                        face_bbox = face_boxes(frame) # ova linija se sporo izvrsava, ali brze od YOLO modela
                        # if frame_num % 100 == 0:
                        #     print(f"Time taken to process frame {frame_num} with faceBoxes: {time.time() - fb_start_time:.6f} seconds")

                        # ako nije nadjeno lice na frejmu, frejm ne ulazi u rppg, ali ga sacuvaj na disku zbog provjere
                        if len(face_bbox) == 0:
                            print("Nije nadjeno lice na frejmu", frame_num)
                            if not os.path.exists(face_not_found_folder):
                                os.makedirs(face_not_found_folder)
                            if segment_id == 'no_segmentation':
                                face_not_found_image_path = os.path.join(face_not_found_folder, "{0}_T{1}_frame{2}.jpg".format(person, task, frame_num))
                            else:
                                face_not_found_image_path  = os.path.join(face_not_found_folder, "{0}_T{1}_seg{2}_frame{3}.jpg".format(person, task, segment, frame_num))
                            cv2.imwrite(face_not_found_image_path, frame)
                            continue

                        # ako je pronadjeno lice na frejmu
                        if frame_num % 70 == 0: # stampaj svaki 70. da vidis da radi program (70 frames / 35fps = 2 sekunde videa)
                            print("Found face for subject {0}, task {1}, segment {2}, frame {3}".format(person, task, segment, frame_num))

                        frame_face = cropped_img(frame, face_bbox) # -> MatLike

                        # If needed, convert RGB to BGR for OpenCV compatibility --> ovo se ne radi kada je model za face-detection faceBoxes
                        # if frame_face.ndim == 3 and frame_face.shape[2] == 3:
                        #     frame_face = cv2.cvtColor(frame_face, cv2.COLOR_RGB2BGR)
                        
                        frame_face = cv2.resize(frame_face, (128, 128))
                        frame_face = skin_segment(frame_face)

                        # ovdje sacuvaj izlaz iz skin_segment funkcije, samo za prezentaciju kako izgleda slika na izlazu, kada se na originalnu sliku stavi binarna maska

                        batch[count] = frame_face
                        count += 1
                    # print(f"Time taken to process {len(batch_of_frames_bytes)} frames: {time.time() - process_bytes_start_time:.2f} seconds")
                print(f"Time taken to get rppg from video: {time.time() - start_time:.2f} seconds")

                print("Racuna rppg za", person, "task", task, 'segment', segment)
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
                get_rppg(mean_rgb, person, task, segment)

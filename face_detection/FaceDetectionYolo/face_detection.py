from io import BytesIO
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
from PIL import ImageDraw
yoloModel = YOLO("face_detection/FaceDetectionYolo/yolov11n-face.pt")

import cv2
import os

from io import BytesIO
import time


def eval_image(contents):
    original_image = Image.open(contents)
    original_image = ImageOps.exif_transpose(original_image)
    predicted_image = yoloModel(original_image, verbose=False)  # Run the prediction on the image
    if len(predicted_image) == 0:
        return ("Nema lica", original_image)
    for result in predicted_image:
        boxes = result.boxes
        # pretpostavljam da ima samo jedno lice na slici
        if len(result.boxes) == 0:
            return ("Nema lica", original_image)
        x_topleft = int(boxes.xyxy[0][0])
        y_topleft = int(boxes.xyxy[0][1])
        x_bottomright = int(boxes.xyxy[0][2])
        y_bottomright = int(boxes.xyxy[0][3])
        box_width = x_bottomright - x_topleft
        box_height = y_bottomright - y_topleft

        max_size = max(box_width, box_height)
        min_size = min(box_width, box_height)
        diff = (max_size - min_size) // 2
        # ako dimenzije bounding box-a nisu u aspect ratio 1:1
        # dopunjam onu manju dimenziju jednako sa lijeve i desne strane
        # odnosno gore i dolje
        if box_width < box_height:
            x_topleft -= diff
            x_bottomright += diff
            x_topleft = max(0, x_topleft)
            x_bottomright = min(x_bottomright, original_image.size[0])
        else:
            y_topleft -= diff
            y_bottomright += diff
            y_topleft = max(0, y_topleft)
            y_bottomright = min(y_bottomright, original_image.size[1])

    original_image = original_image.crop(
        (x_topleft, y_topleft, x_bottomright, y_bottomright)
    )

    np_array = np.asarray(original_image, dtype=np.float32)
    return np_array

def read_from_path(path:str):
    with open(path, "rb") as file:
        contents = file.read()
    return BytesIO(contents)


if __name__ == "__main__":

    # dataset_root = "/Volumes/externiMAC/UBFC-Phys"
    # dataset = "/Volumes/externiMAC/UBFC-Phys/dataset"
    # havent_done = ['s1', 's2', 's3', 's4', 's5']

    # for person in havent_done:
    #     person_path = os.path.join(dataset, person)
        
    #     for task in [1,2,3]:
    #         video_path = os.path.join(person_path, "vid_{0}_T{1}.avi".format(person, task))
    #         # print(video_path)
    #         vidcap = cv2.VideoCapture(video_path)
    #         success,image = vidcap.read()
    #         count = 0 #  koliko frejmova ce biti obradjeno u face_detection
    #         frame_num=0 # koliko frejmova je procitano
    #         frame_rate = 1000
    #         while success:
    #             if frame_num % frame_rate == 0: #vraca svaki 500. frejm (jer frame_rate == 500)
    #                 count += 1  
    #                 cv2.imwrite(os.path.join(dataset_root, "tmp/tmp.jpg"), image)    # save frame as temporary JPG file
    #                 contents = read_from_path(os.path.join(dataset_root, "tmp/tmp.jpg"))

    #                 # obrada frejma - detekcija lica
    #                 predicted_array = eval_image(contents)
    #                 print("Predicted face in image {0}/T{1}/frame_{2}_.jpg".format(person, task, count))

    #                 face_image = Image.fromarray(predicted_array.astype(np.uint8))
    #                 face_image.save(os.path.join(dataset_root, "tmp_rppg/{0}.jpg".format(count)))

    #             success,image = vidcap.read()
    #             frame_num += 1
            
    #         #TODO
    #         # u tmp_rppg se nalaze fokusirane slike lica, sad nad tim folderom treba da se pozovu funkcije iz pos_rppg.py
    #         # ovdje ide kod, izvan while petlje, pocinje main iz fajla pos_rppg.py
    #         print("Racuna rppg za", person, "task", task)
    #         # folder gdje se nalaze slike detektovanih faca
    #         faces_path = os.path.join(dataset_root, "tmp_rppg")
    #         faces_files_names = os.listdir(faces_path) # imena slika

    #         # jer mi kreira .DS_Store file
    #         for fname in faces_files_names:
    #             if fname.startswith("."):
    #                 faces_files_names.remove(fname)

    #         # sortira da bi vidio koliko ih ima ukupno
    #         faces_files_names.sort(key=lambda x: int(x.split('.')[0]))
    #         total_lengh = int(faces_files_names[-1].split('.')[0])
    #         print("broj slika:", total_lengh)

    #-------------------------------------------------------------------------------------

    # dir = "frames/s1/T1"
    # frames = os.listdir(dir)
    # path = "frames/s1/T1/frame500.jpg"
    
    # for path in frames:
    #     frame_num = int(path.split("_")[1])
    #     contents = read_from_path(os.path.join(dir, path))
    #     predicted_array = eval_image(contents)
    #     image = Image.fromarray(predicted_array.astype(np.uint8))
        
    #     # image.save("/Volumes/externiMAC/UBFC-Phys/s1/T1/face/%d.jpg" % frame_num)
    #     print("saved image", frame_num)
    #-------------------------------------------------------------------------------------

    vidcap = cv2.VideoCapture('/Volumes/stari/UBFC-Phys/segment_30s/dataset/s1/vid_s1_T1_seg03.avi')
    success, img = vidcap.read()
    if not success:
        print('Failed to read image from video')
    
    success, encoded_frame = cv2.imencode(".png", img)
    if success: # ovaj dio koda se brzo izvrsava
        frame_bytes = encoded_frame.tobytes()
        contents = BytesIO(frame_bytes)
    else:
        raise ValueError("Could not encode frame to image format.")

    start_time = time.time()
    for i in range(1):
        predicted_array = eval_image(contents)
        if i % 20 == 0:
            print(i, "Predicted face in image")
    print(f'Inference time for 500 iterations: {time.time() - start_time:.2f}s') # --> oko 40 sekundi
    print(type(predicted_array))  # (h, w, 3)

    img = Image.fromarray(predicted_array.astype(np.uint8))
    img.save('face_detection/cropped_face_yolo.jpg')
    # cropped_img = Image.open(predicted_array)
    # cropped_img.save('face_detection/cropped_face_yolo.jpg')

    
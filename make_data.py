import os
from mtcnn import MTCNN
import cv2
import dlib
import pickle
from imutils import face_utils

detector = MTCNN()
landmark_detector = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# 1. Đọc thư mục face_data

raw_folder = "face_data"

landmark_list = []
label_list = []

for folder in os.listdir(raw_folder):
    if folder[0]!=".":
        print("Process folder ", folder)

        for file in os.listdir(os.path.join(raw_folder,folder)):
            print("Process file ", file)

            # Phát hiện khuôn mặt (Coi như 1 ảnh chỉ có 1 khuôn mặt)
            pix_file = os.path.join(raw_folder,folder,file)
            image = cv2.imread(pix_file)

            results = detector.detect_faces(image)

            if len(results)>0:
                # Có mặt, lấy mặt đầu tiên
                result = results[0]

                # Trích xuất toạ độ của mặt trong ảnh
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2 = x1 + width
                y2 = y1 + height

                face = image[y1:y2, x1:x2]

                # Trích xuất landmark băng dlib
                landmark = landmark_detector(image, dlib.rectangle(x1,y1,x2,y2))
                landmark = face_utils.shape_to_np(landmark)

                landmark = landmark.reshape(68*2)


                # Thêm cái landmark vào list các landmark
                landmark_list.append(landmark)
                label_list.append(folder)

print(len(landmark_list))
# Chuyển sang numpy array
import numpy as np
landmark_list = np.array(landmark_list)
label_list = np.array(label_list)

# Write file landmark.pkl
file = open("landmarks.pkl",'wb')
pickle.dump(landmark_list,file)
file.close()

# Write file landmark.pkl
file = open("labels.pkl",'wb')
pickle.dump(label_list,file)
file.close()



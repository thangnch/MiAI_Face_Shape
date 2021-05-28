import pickle
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import dlib
import os
from mtcnn.mtcnn import MTCNN
from random import random
import cv2
from imutils import face_utils
import numpy as np
import  sys


# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

filename = 'model.sav'
clf = pickle.load(open(filename, 'rb'))

desc_file = "face_desc.csv"
f = open(desc_file, "r")
desc = f.readlines()
f.close()
dict = {}
for line in desc:
    dict[line.split('|')[0]] = [line.split('|')[1],line.split('|')[2]]

detector = MTCNN()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                print(image.filename)
                print(app.config['UPLOAD_FOLDER'])
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                results = detector.detect_faces(frame)

                if len(results)!=0:
                    for result in results:
                        x1, y1, width, height = result['box']

                        x1, y1 = abs(x1), abs(y1)
                        x2, y2 = x1 + width, y1 + height
                        face = frame[y1:y2, x1:x2]

                        # Extract dlib
                        landmark = predictor(frame, dlib.rectangle(x1, y1, x2, y2))
                        landmark = face_utils.shape_to_np(landmark)

                        print("O", landmark.shape)
                        landmark = landmark.reshape(68 * 2)
                        print("R", landmark.shape)

                        y_pred = clf.predict([landmark])
                        print(y_pred)

                        extra = dict[y_pred[0]][1]
                        ID = dict[y_pred[0]][0]

                        cv2.imwrite(path_to_save, face)
                        break

                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", idBoolean = True, ID=ID, extra=extra)
                else:
                    return render_template('index.html', msg='Không nhận diện được khuôn mặt')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được khuôn mặt')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True)
    run_with_ngrok(app)   
    app.run()

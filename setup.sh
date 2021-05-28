pip install -r requirements.txt
mkdir models
wget -P ./models/ http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bunzip2 ./shape_predictor_68_face_landmarks.dat.bz2
wget https://drive.google.com/file/d/16XGXgfkihNsUnxbZyWmYWwmfG7sGb9u1/view?usp=sharing &&  cp ./face_shape_data/face_data .
import pickle
from sklearn import svm

# Load dữ liệu từ 2 file pkl

file = open('landmarks.pkl','rb')
landmark_list = pickle.load(file)
file.close()

file = open('labels.pkl','rb')
label_list = pickle.load(file)
file.close()

svm = svm.SVC(kernel='linear')  # SVM param gridsearch
svm.fit(landmark_list, label_list) # Train

# Thử predict mặt đầu tiên
result = svm.predict([landmark_list[0]])
print("Kết quả predict ", result, " Giá trị thực = ", label_list[0])

# Lưu model vào file
model_file = "model.sav"
file = open(model_file,'wb')
pickle.dump(svm, file)
file.close()

# Load xong thì train model SVM
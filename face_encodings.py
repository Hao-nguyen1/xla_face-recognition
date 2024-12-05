import face_recognition
import pickle
import os

# Đường dẫn đến thư mục chứa ảnh
image_dir = "pic"

# Dictionary để lưu tên và face encodings
face_db = {}

for filename in os.listdir(image_dir):
    path = os.path.join(image_dir, filename)
    # Chỉ xử lý các tệp hình ảnh (jpg, png, jpeg)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        # Nếu tìm thấy face encoding thì lưu
        if encodings:
            face_db[filename.split(".")[0]] = encodings[0]

# Lưu face encodings vào tệp
with open("face_database.pkl", "wb") as f:
    pickle.dump(face_db, f)

print("Cơ sở dữ liệu khuôn mặt đã được tạo và lưu vào face_database.pkl")

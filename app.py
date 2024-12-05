import streamlit as st
import face_recognition
import cv2
import tempfile
import pickle
import os
import json
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import cvzone
from ultralytics import YOLO


with open("face_database.pkl", "rb") as f:
    face_db = pickle.load(f)

with open("person_info.json", "r") as f:
    person_info = json.load(f)

def draw_text_with_pil(image, text, position, font_path="arial.ttf", font_size=24, color=(0, 255, 0)):
    """
    Hiển thị văn bản tiếng Việt trên ảnh OpenCV sử dụng PIL.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def recognize_face_from_frame(frame, id_encoding=None):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    recognized_names = []

    # Lặp qua từng mã hóa đặc trưng trong khung hình
    for encoding in face_encodings:
        recognized = False  # Đánh dấu xem khuôn mặt hiện tại đã được nhận diện hay chưa

        # So sánh với các khuôn mặt trong cơ sở dữ liệu (face_db)
        for name, db_encoding in face_db.items():
            matches = face_recognition.compare_faces([db_encoding], encoding)
            if True in matches:  
                if id_encoding is not None:
                    id_match = face_recognition.compare_faces([id_encoding], encoding)
                    if id_match[0]: 
                        recognized_names.append(name)  
                    else:  
                        recognized_names.append("ID Mismatch")  
                else:
                    recognized_names.append(name) 
                recognized = True 
                break  
        
        if not recognized:
            recognized_names.append("Unknown")  
    return recognized_names, face_locations

def extract_face_and_info_from_cccd(image_path):
    """
    Trích xuất mã hóa khuôn mặt và thông tin từ ảnh CCCD.
    """
    # Trích xuất khuôn mặt từ ảnh
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    if not face_encodings:
        return None, "Không tìm thấy khuôn mặt trong ảnh CCCD."

    face_encoding = face_encodings[0]  # Lấy khuôn mặt đầu tiên
    
    # Trích xuất thông tin bằng EasyOCR
    cccd_info = extract_info_from_cccd_easyocr(image_path)
    
    return face_encoding, cccd_info

def extract_info_from_cccd_easyocr(image_path):
    """
    Trích xuất thông tin từ ảnh CCCD bằng EasyOCR.
    """
    # Khởi tạo EasyOCR với ngôn ngữ tiếng Việt
    reader = easyocr.Reader(['vi'])  # Hỗ trợ tiếng Việt
    result = reader.readtext(image_path, detail=0)  # Trích xuất văn bản từ ảnh
    
    # In ra kết quả raw để kiểm tra
    print("OCR Raw Result:", result)

    text = "\n".join(result)  # Gộp tất cả các dòng văn bản

    # Tìm kiếm thông tin quan trọng
    info = {
        "Họ và tên": "",
        "Số CCCD": "",
        "Quê quán": ""
    }

    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "Full name" in line or "Ho va ten" in line:
            if i + 1 < len(lines):
                info["Họ và tên"] = lines[i + 1].strip()
        elif "Citizen Identity Card" in line or "No" in line:
            if i + 1 < len(lines):
                info["Số CCCD"] = lines[i + 1].strip()
        elif "Place of origin" in line or "Quê quán" in line:
            if i + 1 < len(lines):
                info["Quê quán"] = lines[i + 1].strip()


    return info



model = YOLO("models/l_version_1_300.pt").to("cpu")

classNames = ["fake", "real"]

st.title("Xác thực Khuôn mặt")

# Tùy chọn xác thực
input_method = st.radio(
    "Chọn phương thức xác thực",
    ("Bước 1: Tải ảnh CCCD", "Bước 2: Xác thực bằng webcam", "Bước 3: Xác thực KYC qua webcam", 
     "Xác thực bằng video tải lên", "Bước cuối: Liveness")
)

warning_displayed = False

# Xử lý chế độ "Tải ảnh CCCD"
if input_method == "Bước 1: Tải ảnh CCCD":
    uploaded_image = st.file_uploader("Tải lên ảnh CCCD", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            temp_image_file.write(uploaded_image.read())
            temp_image_path = temp_image_file.name

        # Hiển thị ảnh đã tải lên
        st.image(temp_image_path, caption="Ảnh CCCD đã tải lên", use_column_width=True)

        # Trích xuất khuôn mặt và thông tin từ ảnh CCCD
        with st.spinner("Đang xử lý ảnh CCCD..."):
            face_encoding, cccd_info = extract_face_and_info_from_cccd(temp_image_path)

        if face_encoding is None:
            st.error(cccd_info)
        else:
            st.success("Khuôn mặt và thông tin đã được trích xuất.")
            st.subheader("Thông tin trích xuất:")
            for key, value in cccd_info.items():
                st.write(f"**{key}**: {value}")
            
            # Lưu mã hóa khuôn mặt và thông tin CCCD vào file
            saved_data = {
                "face_encoding": face_encoding.tolist(),
                "cccd_info": cccd_info  # Lưu thông tin CCCD đầy đủ
            }
            with open("cccd_data.pkl", "wb") as f:
                pickle.dump(saved_data, f)
            st.success("Thông tin đã được lưu vào file `cccd_data.pkl`.")

        os.remove(temp_image_path)



elif input_method == "Xác thực bằng video tải lên":
    # Phần mã hiện tại của bạn về xử lý video tải lên
    uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "mov", "avi"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_path = temp_video_file.name

        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            recognized_names, face_locations = recognize_face_from_frame(frame)

            for (top, right, bottom, left), name in zip(face_locations, recognized_names):
                if name in person_info:
                    # Lấy thông tin từ person_info
                    email = f"Email: {person_info[name]['email']}"
                    phone = f"Phone: {person_info[name]['phone']}"
                    address = f"Address: {person_info[name]['address']}"
                else:
                    email, phone, address = "Unknown", "", ""

                # Vẽ hình chữ nhật xung quanh khuôn mặt
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1  # Tăng cỡ chữ
                color = (0, 0, 255)
                thickness = 2

                # Hiển thị từng dòng văn bản ở rìa bên phải của khung
                cv2.putText(frame, name, (right + 10, top - 50), font, 1.5, color, thickness)
                cv2.putText(frame, email, (right + 10, top - 20), font, font_scale, color, thickness)
                cv2.putText(frame, phone, (right + 10, top + 20), font, font_scale, color, thickness)
                cv2.putText(frame, address, (right + 10, top + 50), font, font_scale, color, thickness)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame)

        cap.release()
        os.remove(temp_video_path)


elif input_method == "Bước 2: Xác thực bằng webcam":
    if not os.path.exists("cccd_data.pkl"):
        st.warning("Bạn cần tải ảnh CCCD trước khi sử dụng tính năng này.")
    else:
        # Nạp dữ liệu từ file
        with open("cccd_data.pkl", "rb") as f:
            saved_data = pickle.load(f)
            cccd_face_encoding = saved_data["face_encoding"]
            cccd_info = saved_data["cccd_info"]

        st.warning("Nhấn 'Start' để bắt đầu xác thực qua webcam.")
        start_webcam = st.button("Start")
        stop_webcam = st.button("Stop")

        if start_webcam:
            cap = cv2.VideoCapture(1)  # Khởi động webcam
            # Bắt đầu xác thực bằng webcam
            cap = cv2.VideoCapture(1)
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # So sánh với khuôn mặt trong file CCCD
                    matches = face_recognition.compare_faces([cccd_face_encoding], face_encoding)
                    
                    if matches[0]:
                        label = "Khớp với CCCD"
                        color = (0, 255, 0)  # Màu xanh lá cây

                        # Lấy thông tin từ CCCD
                        name = cccd_info.get("Họ và tên", "Không rõ")
                        cccd_number = cccd_info.get("Số CCCD", "Không rõ")

                        # Hiển thị thông tin trên khung hình
                        frame = draw_text_with_pil(frame, f"Tên: {name}", (left, bottom + 20), "arial.ttf", 24, color)
                        frame = draw_text_with_pil(frame, f"Số CCCD: {cccd_number}", (left, bottom + 45), "arial.ttf", 24, color)
                    else:
                        label = "Không khớp"
                        color = (0, 0, 255)  # Màu đỏ

                    # Vẽ hình chữ nhật xung quanh khuôn mặt và hiển thị label
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    frame = draw_text_with_pil(frame, label, (left, top - 10), "arial.ttf", 24, color)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame)

            cap.release()


elif input_method == "Bước 3: Xác thực KYC qua webcam":

    st.warning("Nhấn 'Start' để bắt đầu xác thực KYC qua webcam.")
    start_webcam = st.button("Start")
    stop_webcam = st.button("Stop")

    if start_webcam:
        cap = cv2.VideoCapture(1)  # Khởi động webcam
        cap = cv2.VideoCapture(1)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if len(face_encodings) >= 2:
                face_areas = [(bottom - top) * (right - left) for (top, right, bottom, left) in face_locations]
                main_face_index = face_areas.index(max(face_areas))
                main_face_encoding = face_encodings[main_face_index]

                for i, (face_encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
                    if i == main_face_index:
                        label = "Nguoi dung"
                        color = (0, 255, 0)
                    else:
                        match = face_recognition.compare_faces([main_face_encoding], face_encoding)
                        label = "Khop voi nguoi dung" if match[0] else "khong khop"
                        color = (0, 255, 0) if match[0] else (0, 0, 255)

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label, (left, top - 10), font, 0.7, color, 2)

                warning_displayed = False

            else:
                if not warning_displayed:
                    st.warning("Vui lòng cầm ảnh khuôn mặt của bạn trước webcam để xác thực.")
                    warning_displayed = True

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame)

        cap.release()

elif input_method == "Bước cuối: Liveness":
    st.header("liveness")
    
    confidence_threshold = st.slider("Ngưỡng độ tin cậy", 0.0, 1.0, 0.2, 0.01)
    source_option = st.radio("nguồn", ("Webcam"))
    
    if source_option == "Webcam":
        cap = cv2.VideoCapture(1)
    else:
        uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "mov", "avi"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(uploaded_file.read())
                video_path = temp_video_file.name
            cap = cv2.VideoCapture(video_path)
        else:
            st.warning("Vui lòng tải lên video để tiếp tục.")
            cap = None
    
    if cap:
        stframe = st.empty()
        prev_frame_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            new_frame_time = time.time()
            results = model(frame, stream=True, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    if conf > confidence_threshold:
                        color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cvzone.putTextRect(
                            frame, f'{classNames[cls].upper()} {int(conf * 100)}%',
                            (max(0, x1), max(35, y1)), scale=2, thickness=4,
                            colorR=color, colorB=color
                        )
            
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        if source_option == "Tải video":
            os.remove(video_path)

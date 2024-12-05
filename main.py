import math
import time

import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.2

cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video


model = YOLO("models/l_version_1_300.pt")
print(model.names)
classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]  # Không cần làm tròn
            cls = int(box.cls[0])
            print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, conf: {conf}, class: {cls}")
            # Vẽ bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if conf > confidence:

                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO("../yolo_weights/yolov8l.pt")

# Use a set to store the classes of interest
classNames = ["cell phone"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    cell_phone_detected = False  # Flag to check if cell phone is detected
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Check if the detected object is in the set of classes of interest
            if 0 <= cls < len(classNames) and classNames[cls] == "cell phone":
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class name
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))
                cell_phone_detected = True  # Set the flag to True

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Beep when a cell phone is detected
    if cell_phone_detected:
        winsound.Beep(1000, 500)

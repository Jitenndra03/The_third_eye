import cv2
import numpy as np

# Load pre-trained MobileNetSSD model for object detection
net = cv2.dnn.readNetFromCaffe("C:\Users\Dell\Python\Project\MobileNetSSD_deploy.prototxt", "C:\Users\Dell\Python\Project\MobileNetSSD_deploy.caffemodel")

def detect_mobile_phone(frame):
    # Resize frame to have a maximum width of 600 pixels (for faster processing)
    frame_resized = cv2.resize(frame, (600, 600))
    height, width = frame_resized.shape[:2]

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above a certain threshold (e.g., 20%)
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])

            # MobileNetSSD class label for "cell phone" is 77
            if class_id == 77:
                # Calculate bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame_resized, "Mobile Phone", (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame_resized

# Capture video from webcam (you can also read from a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect mobile phones in the frame
    result_frame = detect_mobile_phone(frame)

    # Display the resulting frame
    cv2.imshow("Mobile Phone Detection", result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

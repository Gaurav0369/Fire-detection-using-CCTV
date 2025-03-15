import cv2
import os
from ultralytics import YOLO

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)
threshold = 0.5

def process_video():
    capture = cv2.VideoCapture(0) # Use 0 for webcam, or replace with video file path
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        results = model(frame)[0]
        # Process the frame with the YOLO model and yield the processed frame
        yield frame
        
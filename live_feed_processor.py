import cv2
import numpy as np
from ultralytics import YOLO

input = 0
class LiveFeedHandler:
    def __init__(self, model_path, threshold=0.5, resize_percentage=0.1):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.resize_percentage = resize_percentage
        self.capture = cv2.VideoCapture(input) # Assuming default camera for now

    def process_frame(self, frame):
        # Calculate the new dimensions
        height, width = frame.shape[:2]
        new_width = int(width * self.resize_percentage)
        new_height = int(height * self.resize_percentage)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Process the resized frame with the YOLO model
        results = self.model(resized_frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > self.threshold:
                # Convert coordinates to the original frame size
                x1 = int(x1 * (1 / self.resize_percentage))
                y1 = int(y1 * (1 / self.resize_percentage))
                x2 = int(x2 * (1 / self.resize_percentage))
                y2 = int(y2 * (1 / self.resize_percentage))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        return frame

    def generate_frames(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            processed_frame = self.process_frame(frame)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which is used by images on the web)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

    # Add this method to the LiveFeedHandler class in live_feed_processor.py
    # Adjusted method in live_feed_processor.py
    # Adjusted method in live_feed_processor.py
    def detect_fire(self, frame):
        # Assuming 'predict' is the correct method for performing inference with the Ultralytics YOLO model
        # and 'FIRE_CLASS_ID' is the class ID for fire in your model
        results = self.model.predict(frame)
        # Assuming results is a list of confidence scores for each class
        # Find the index of the highest confidence score
        max_conf_index = results.index(max(results))
    # Check if the class ID corresponding to the highest confidence score is for fire
        if max_conf_index == (0 or 1) and results[max_conf_index] > self.threshold:
            return True
        return False



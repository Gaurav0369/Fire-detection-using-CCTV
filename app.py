from flask import Flask, Response, render_template
from live_feed_processor import LiveFeedHandler
import cv2
from ultralytics import YOLO


import os


app = Flask(__name__)

# YOLO model initialization
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
# Assuming the model path is stored in a variable named 'model_path'
live_feed_handler = LiveFeedHandler(model_path)

model = YOLO(model_path)

# Initialize video capture
capture = cv2.VideoCapture(0) # Assuming default camera for now
# Global flag to control the loop
stop_loop = False


def video_feed_generator():
    global stop_loop
    while not stop_loop:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame)[0]

        # Check if any object is detected with a confidence score above the threshold
        detected = any(score > 0.5 for score in results.scores.data.tolist())

        # Yield the detection status
        yield detected


# Create an instance of LiveFeedHandler
# Replace 'path/to/your/model.pt' with the actual path to your YOLO model
live_feed_handler = LiveFeedHandler(model_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(live_feed_handler.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add this route to app.py
@app.route('/alert')
def alert():
    # Capture a frame from the video feed
    ret, frame = capture.read()
    if not ret:
        return {'error': 'Failed to capture frame.'}
    
    # Use the live_feed_handler instance to detect fire in the captured frame
    fire_detected = live_feed_handler.detect_fire(frame)
    return {'fire_detected': fire_detected}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
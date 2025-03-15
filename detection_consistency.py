import time

# Placeholder for frame timestamps and detection results
frame_timestamps = {}

def is_anything_detected(threshold, max_seconds=5):
    current_time = time.time()
    # Filter out detections that are older than max_seconds
    recent_detections = [timestamp for timestamp in frame_timestamps if current_time - timestamp <= max_seconds]
    
    # Check if there are any detections in the recent_detections list
    return len(recent_detections) > 0

def update_detection(class_id, score, timestamp):
    # Update frame_timestamps with the detection result
    # This is a simplified example; you'll need to adapt it to your specific use case
    frame_timestamps[timestamp] = {'class_id': class_id, 'score': score}

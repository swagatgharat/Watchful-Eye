import cv2 as cv
import numpy as np
import time
import geocoder
import os
from datetime import datetime
import winsound

# This script uses OpenCV and YOLOv4 for real-time pothole detection via a camera.
# It captures video, detects potholes, classifies their severity (Low, Medium, High),
# and saves detection results (images and severity information) with geolocation data.
# The program runs in full-screen mode and allows quitting with the 'q' key.

cap = None  # Initialize video capture
result = None  # Initialize video writer for saving output video

try:
    # Get the current time for naming saved files
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Load class names for detection from a file (e.g., 'pothole')
    class_name = []
    with open(r'C:\Users\Swagat Gharat\Desktop\PotHole\Utils\obj.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # Load the YOLOv4 model configuration and weights
    net1 = cv.dnn.readNet(r'C:\Users\Swagat Gharat\Desktop\PotHole\Utils\yolov4_tiny.weights',
                          r'C:\Users\Swagat Gharat\Desktop\PotHole\Utils\yolov4.tiny.cfg')
    # Set the backend and target for CUDA for faster processing
    net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    # Create a model from the loaded network
    model1 = cv.dnn_DetectionModel(net1)
    model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

    # Initialize video capture from the default camera (index 0)
    cap = cv.VideoCapture(0)

    # Capture the first frame to check if video capture is successful
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to Capture Video")

    # Get the width and height of the captured frame
    width = cap.get(3)
    height = cap.get(4)

    # Create a directory for saving results if it doesn't exist
    result_path = "Pothole Images & Coordinates"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Set up video writer for saving the output video
    result_filename = os.path.join(result_path, f'result_{current_time}.avi')
    result = cv.VideoWriter(result_filename, cv.VideoWriter_fourcc(*'MJPG'), 10, (int(width), int(height)))

    # Get geolocation data (latitude and longitude) of the user
    g = geocoder.ip('me')

    # Initialize timing variables
    starting_time = time.time()
    Conf_threshold = 0.5  # Confidence threshold for detection
    NMS_threshold = 0.4    # Non-Maximum Suppression threshold
    frame_counter = 0      # Frame counter for calculating FPS
    i = 0                  # Counter for saved images
    b = 0                  # Time tracker for saving images

    # Create a mask to focus on the relevant part of the frame
    mask = np.zeros_like(frame)
    mask[0:int(0.85 * height), :] = 255  # Keep upper 85% of the frame for detection

    # Set up a full-screen window for display
    window_name = 'frame'
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Define colors for severity levels
    severity_colors = {
        "Low": (0, 255, 255),      # Yellow
        "Medium": (0, 165, 255),   # Orange
        "High": (0, 0, 255)        # Red
    }

    # Define beep frequencies for severity levels
    severity_frequencies = {
        "Low": 1000,      # Frequency for low severity
        "Medium": 2000,   # Frequency for medium severity
        "High": 3000      # Frequency for high severity
    }

    while True:
        try:
            # Capture a frame from the camera
            ret, frame = cap.read()
            frame_counter += 1  # Increment the frame counter
            if not ret:
                break  # Break loop if frame capture fails

            # Apply the mask to the captured frame
            masked_frame = cv.bitwise_and(frame, mask)

            # Perform object detection on the masked frame
            classes, scores, boxes = model1.detect(masked_frame, Conf_threshold, NMS_threshold)

            # Process each detected object
            for (classid, score, box) in zip(classes, scores, boxes):
                label = "Pothole"  # Set label for detected object
                x, y, w, h = box  # Get bounding box coordinates
                recarea = w * h  # Calculate the area of the detection box
                area = width * height  # Calculate the area of the entire frame

                severity = ""  # Initialize severity classification
                # Define thresholds for severity classification
                severity_threshold_low = 0.01
                severity_threshold_medium = 0.02
                severity_threshold_high = 0.05  # Adjusted for better classification

                # Check if detection score is above the threshold
                if len(scores) != 0 and scores[0] >= 0.5:
                    # Classify severity based on the area of the detection box
                    if (recarea / area) <= severity_threshold_low:
                        severity = "Low"
                    elif (recarea / area) <= severity_threshold_medium:
                        severity = "Medium"
                    else:
                        severity = "High"

                    if severity != "":
                        # Draw rectangle around detected pothole
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        # Get color for the severity level
                        severity_color = severity_colors.get(severity, (255, 255, 255))  # Default to white if not found
                        # Put text indicating detection confidence and severity
                        cv.putText(frame, f"%{round(scores[0] * 100, 2)} {label} ({severity} Severity)",
                                   (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, severity_color, 1)

                        # Beep according to severity level
                        beep_frequency = severity_frequencies[severity]
                        winsound.Beep(beep_frequency, 100)  # Beep for 100 milliseconds

                        # Save detection results with timestamps
                        if i == 0:
                            cv.imwrite(os.path.join(result_path, f'pot_{current_time}_{i}.png'), frame)
                            with open(os.path.join(result_path, f'pot_{current_time}_{i}.txt'), 'w') as f:
                                f.write(f"{str(g.latlng)}\nSeverity: {severity}")  # Save geolocation and severity
                            i += 1  # Increment image counter

                        if i != 0:
                            if (time.time() - b) >= 2:  # Save every 2 seconds
                                cv.imwrite(os.path.join(result_path, f'pot_{current_time}_{i}.png'), frame)
                                with open(os.path.join(result_path, f'pot_{current_time}_{i}.txt'), 'w') as f:
                                    f.write(f"{str(g.latlng)}\nSeverity: {severity}")  # Save geolocation and severity
                                b = time.time()  # Update the last save time
                                i += 1  # Increment image counter

            # Calculate and display FPS
            ending_time = time.time() - starting_time  # Calculate elapsed time
            fps = frame_counter / ending_time  # Calculate frames per second

            # Define the text for displaying FPS and its rectangle size
            fps_text = f'FPS: {fps:.2f}'
            text_size = cv.getTextSize(fps_text, cv.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
            padding = 10
            # Draw a background rectangle for FPS text
            cv.rectangle(frame, (20 - padding, 50 - text_size[1] - padding),
                          (20 + text_size[0] + padding, 50 + padding), (0, 0, 0), cv.FILLED)
            cv.putText(frame, fps_text, (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)  # Display FPS

            # Show the current frame in the window
            cv.imshow(window_name, frame)
            result.write(frame)  # Write the current frame to the video file

            # Check for user input to quit
            key = cv.waitKey(1)
            if key == ord('q'):
                break  # Exit loop on 'q'

        except Exception as e:
            print(f"Error: {e}")  # Print any errors that occur

except Exception as e:
    print(f"Error: {e}")  # Print any errors during setup

finally:
    # Release resources and close windows
    if cap is not None:
        cap.release()  # Release the video capture object
    if result is not None:
        result.release()  # Release the video writer object
    cv.destroyAllWindows()  # Close all OpenCV windows

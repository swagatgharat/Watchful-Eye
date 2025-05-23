Watchful Eye Project
This repository contains the code for a real-time pothole detection system using OpenCV and YOLOv4.
The project captures video from a camera, detects potholes, classifies their severity, and saves detection results along with geolocation data.
It provides an interactive experience and is designed for both functionality and ease of use.

Features:
1. Real-Time Pothole Detection:
Object Detection: Utilizes YOLOv4 for accurate detection of potholes in video feeds.
Severity Classification: Classifies detected potholes into three severity levels: Low, Medium, and High.

2. Geolocation Tracking:
Captures and saves geolocation data (latitude and longitude) for each detected pothole.

3. Image and Video Output:
Saves detection results, including images and text files with severity information and geolocation data.
Outputs a video file of the detection process.

4. Audio Alerts:
Provides audible alerts (beeps) based on the severity of detected potholes.

5. Performance Monitoring:
Displays the frames per second (FPS) to monitor system performance during detection.

Technologies:
Programming Language: Python

Libraries:
1. OpenCV for computer vision tasks
2. NumPy for numerical operations
3. Geocoder for geolocation
4. Winsound for audio alerts

Uses:
1. Road Maintenance:
Helps municipal authorities identify and prioritize pothole repairs, improving road safety and infrastructure.

2. Data Collection:
Assists in gathering data on pothole locations and severity, useful for research and urban planning.

3. Public Safety:
Enhances the safety of road users by providing timely alerts about hazardous road conditions.

4. Community Engagement:
Empowers citizens to report pothole issues through community-driven monitoring initiatives.

4. Automated Inspection:
Can be integrated into existing road inspection systems to automate the detection process, saving time and resources.

5. In-Car Detection:
Facilitates real-time pothole detection directly from a vehicle, allowing for proactive identification of road hazards while driving.

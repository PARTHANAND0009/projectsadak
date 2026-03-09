# 🚧 Sadak AI — Automated Pothole Detection System

Detects potholes in real-time using a webcam, marks them with bounding boxes, 
and automatically emails authorities with photo + location.

## Built With
- YOLOv8 (custom trained on 500+ pothole images)
- OpenCV (live webcam feed)
- Python + smtplib (automated Gmail alerts)

## How It Works
1. Webcam scans the road in real-time
2. AI model detects potholes instantly
3. Screenshot saved with bounding box
4. Email sent automatically to authorities

## Setup
```bash
pip install ultralytics opencv-python
python main.py
```

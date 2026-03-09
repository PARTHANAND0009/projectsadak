import cv2
import os
import smtplib
import datetime
from email.mime.text import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from ultralytics import YOLO

# Load the YOLOv8 model
MODEL_PATH = 'runs/detect/train/weights/best.pt'  # Update with your model path

SENDER_EMAIL = 'parthanand67@gmail.com'  # Update with your email
SENDER_PASSWORD = '24227PrimeMedia'  # Update with your email password for the sender email id
RECEIVER_EMAIL = 'mcd-ithelpdesk@mcd.nic.in'  # Update with the receiver's email

CONFINDENCE_THRESHOLD = 0.5  # Confidence threshold for detections
COOLDOWN_SECONDS = 10  # Cooldown period in seconds to avoid spamming emails
SAVE_FOLDER = 'DETECTIONS'  # Folder to save detected images of the potholes

def send_alert_email(image_path, location, timestamp):
    '''Sends an email alert with the detected image attached.'''
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = f'Alert: Pothole Detected at {location}'
        body = f'''
Dear Road Authority,

A pothole has been detected at the following location:

Location: {location}
Timestamp: {timestamp}

Please find the attached image for reference. Kindly make arrangements for repair at the earliest to ensure road safety.

This is an automated message from Team Sadak AI.
'''        
       msg.attach(MIMEText(body, 'plain')) 
    
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(image_path)}")
        msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com',465) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print(f"Alert email sent for {location} at {timestamp}")
    return True

def run_detection():
    '''Runs the pothole detection on the given video.'''
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Starting video capture...")

    location = input("\nEnter the location of the road being monitored: ")
    if not location:
        location = "Unknown Location"

    cap = cv2.VideoCapture(0)  # Update with your video path
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened. Press Q to quit.\n")
    return

    last_alert_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        results = model(frame, conf=CONFINDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        if pothole_count > 0:
            now = datetime.datetime.now()
            elapsed = (now - last_alert_time.fromtimestamp(last_alert_time)).total_seconds() if last_alert_time else 999

            cv2.putText(annotated, f'Potholes Detected: {pothole_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if elapsed > COOLDOWN_SECONDS:
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                img_path = os.path.join(SAVE_FOLDER, f'pothole_{timestamp}.jpg')

                cv2.imwrite(img_path, frame)
                print(f"Pothole detected at {location} at {timestamp}. Image saved to {img_path}")

                send_alert_email(img_path, location, now.strftime("%Y-%m-%d %H:%M:%S"))
                last_alert_time = now.timestamp()

        else:
            cv2.putText(annotated, 'No Potholes Detected', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

            cv2.imshow('Pothole Detector', annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and windows closed."

if __name__ == "__main__":
    run_detection()
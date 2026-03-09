

from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 nano model (fast, good for laptops)
    model = YOLO("yolov8n.pt")  # downloads automatically on first run

    # Train on your dataset
    results = model.train(
        data="dataset/data.yaml",   # path to your dataset config
        epochs=50,                  # increase to 100 for better accuracy
        imgsz=640,                  # image size
        batch=2,                    # reduce to 4 if your PC is slow
        name="pothole_detector",    # folder name for saved model
        patience=10,                # stop early if no improvement
        device="cpu"                # change to 0 if you have a GPU
    )

    print("\n✅ Training complete!")
    print("Your model is saved at: runs/detect/pothole_detector/weights/best.pt")
    print("Use best.pt in detect_and_alert.py")

if __name__ == "__main__":
    train_model()
from ultralytics import YOLO

# Load model
model = YOLO("C:\catan_universe_project\catan_helper\yolo11m_detection.pt")  # Pretrained model

# Process video and save annotated output
results = model.track("C:\catan_universe_project\catan_dataset\Catan game 4 split.mp4", save=True)  # For tracking + detection

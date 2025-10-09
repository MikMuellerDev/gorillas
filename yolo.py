from ultralytics import YOLO

E = 70

import wandb
wandb.init(project="YOLOv11_WB", name="custom_run", config={"epochs": E, "imgsz": 640})

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n.pt")

# Train with W&B tracking enabled
model.train(
    data="yolo.yml",     
    epochs=E,
    imgsz=640,
    tracker="wandb",       
    project="YOLOv11_WB", # optional: custom project name
    name="experiment_1",  # optional: run name
    verbose=True,
    lr0=0.0000001,
    lrf=0.00001,
)

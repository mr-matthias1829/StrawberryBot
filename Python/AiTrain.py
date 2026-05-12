from ultralytics import YOLO
import torch
print(torch.cuda.is_available()) # false = we are using cpu, which is MUCH slower

model = YOLO("yolov8n.pt")   # tiny starter model

model.train(
    data="data.yaml",
    epochs=125,
    imgsz=640,
    batch=16,
    cache="ram",   # loads entire dataset into RAM once, no disk reads after epoch 1
    workers=8,     # parallel data loading threads
    patience=10   # stop if no improvement for 10 epochs
)
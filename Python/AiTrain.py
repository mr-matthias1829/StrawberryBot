from ultralytics import YOLO
import torch
print(torch.cuda.is_available()) # false = we are using cpu, which is MUCH slower

model = YOLO("yolov8n.pt")   # tiny starter model

model.train(
    data="data.yaml",
    epochs=55, # around this point the model seems to hit the peak, more epochs doesn't improve it by much more
    imgsz=320,
    batch=16,
    cache="ram",   # loads entire dataset into RAM once, no disk reads after epoch 1
    workers=8,     # parallel data loading threads
    patience=5   # stop if no improvement for X epochs
)
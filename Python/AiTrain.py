from ultralytics import YOLO
import torch
print(torch.cuda.is_available()) # false = we are using cpu, which is MUCH slower

model = YOLO("yolov8n.pt")   # tiny starter model
#model = YOLO(r"../runs/detect/train-4/weights/best.pt")

model.train(
    data="data.yaml",
    epochs=30, # around this point the model seems to hit the peak, more epochs doesn't improve it by much more
    imgsz=960, # you son of a bi***, IM IN (pls help its gonna take like 5 hours to train this ai aaaaaaa)
    batch=16,
    cache="ram",   # loads entire dataset into RAM once, no disk reads after epoch 1
    workers=8,     # parallel data loading threads
    patience=5   # stop if no improvement for X epochs
)
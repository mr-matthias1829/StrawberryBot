import pandas as pd
import os

# this script is used to convert the "Labels.csv" and images in /dataset into a useable state
# to run: run this exact file. no other python files needed.

CSV_FILE = "Labels.csv"

IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"

os.makedirs(LABELS_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

CLASS_MAP = {
    "Strawberry": 0
}

for image_name, group in df.groupby("image_name"):

    txt_name = image_name.replace(".jpg", ".txt")
    txt_path = os.path.join(LABELS_DIR, txt_name)

    with open(txt_path, "w") as f:

        for _, row in group.iterrows():

            x = row["bbox_x"]
            y = row["bbox_y"]
            w = row["bbox_width"]
            h = row["bbox_height"]

            img_w = row["image_width"]
            img_h = row["image_height"]

            # convert to YOLO format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h

            width = w / img_w
            height = h / img_h

            class_id = CLASS_MAP[row["label_name"]]

            f.write(
                f"{class_id} "
                f"{x_center} "
                f"{y_center} "
                f"{width} "
                f"{height}\n"
            )

print("Conversion complete.")
import pandas as pd
import os

csv_path = "Labels.csv"
output_dir = "labels_txt"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# class mapping
class_map = {
    "Strawberry": 0
}

# group by image (important!)
for image_name, group in df.groupby("image_name"):

    txt_lines = []

    for _, row in group.iterrows():
        img_w = row["image_width"]
        img_h = row["image_height"]

        cls = class_map[row["label_name"]]

        xmin = row["bbox_x"]
        ymin = row["bbox_y"]
        xmax = xmin + row["bbox_width"]
        ymax = ymin + row["bbox_height"]

        # YOLO format
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        txt_lines.append(f"{cls} {x_center} {y_center} {w} {h}")

    # write one file per image
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
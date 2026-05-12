import cv2
import numpy as np
import os
import shutil

# =============================================================================
# CONFIG
# =============================================================================
IMAGES_DIR  = "dataset/images"
LABELS_DIR  = "dataset/labels"
OUT_TRAIN_IMAGES = "dataset_aug/images/train"
OUT_TRAIN_LABELS = "dataset_aug/labels/train"
OUT_VAL_IMAGES   = "dataset_aug/images/val"
OUT_VAL_LABELS   = "dataset_aug/labels/val"

# Toggle augmentations on/off
AUGMENTATIONS = {
    "flip_h":     True,
    "flip_v":     True,
    "rot_15":     True,
    "rot_30":     True,
    "rot_neg15":  True,
    "rot_neg30":  True,
    "brightness": True,
    "hsv_shift":  True,
    "blur":       True,
}

# =============================================================================
# SETUP
# =============================================================================
os.makedirs(OUT_TRAIN_IMAGES, exist_ok=True)
os.makedirs(OUT_TRAIN_LABELS, exist_ok=True)
os.makedirs(OUT_VAL_IMAGES,   exist_ok=True)
os.makedirs(OUT_VAL_LABELS,   exist_ok=True)


# =============================================================================
# LABEL HELPERS
# =============================================================================

def read_labels(label_path: str) -> list[tuple]:
    """Read YOLO .txt → list of (class_id, x_center, y_center, w, h)."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((int(parts[0]), *map(float, parts[1:])))
    return labels


def write_labels(label_path: str, labels: list[tuple]) -> None:
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def save(name: str, img: np.ndarray, labels: list[tuple], train: bool = True) -> None:
    img_dir   = OUT_TRAIN_IMAGES if train else OUT_VAL_IMAGES
    label_dir = OUT_TRAIN_LABELS if train else OUT_VAL_LABELS
    img_path   = os.path.join(img_dir,   name + ".jpg")
    label_path = os.path.join(label_dir, name + ".txt")
    ok = cv2.imwrite(img_path, img)
    if not ok:
        print(f"  [ERROR] Failed to write image: {img_path}")
        return
    write_labels(label_path, labels)


# =============================================================================
# BBOX TRANSFORMS
# =============================================================================

def flip_h(labels: list[tuple]) -> list[tuple]:
    """Horizontal flip: x_center → 1 - x_center."""
    return [(c, 1.0 - xc, yc, w, h) for c, xc, yc, w, h in labels]


def flip_v(labels: list[tuple]) -> list[tuple]:
    """Vertical flip: y_center → 1 - y_center."""
    return [(c, xc, 1.0 - yc, w, h) for c, xc, yc, w, h in labels]


def rotate_labels(
    labels: list[tuple],
    angle_deg: float,
    img_w: int,
    img_h: int,
) -> list[tuple]:
    """Rotate bboxes around image centre; refit axis-aligned bbox after rotation."""
    cx, cy  = img_w / 2, img_h / 2
    theta   = np.radians(-angle_deg)           # cv2 rotates CCW for positive angle
    cos_, sin_ = np.cos(theta), np.sin(theta)

    new_labels = []
    for cls, xc_n, yc_n, w_n, h_n in labels:
        # Denormalise to pixel coords
        xc  = xc_n * img_w
        yc  = yc_n * img_h
        bw  = w_n  * img_w
        bh  = h_n  * img_h

        # All four corners of the box
        corners = np.array([
            [xc - bw / 2, yc - bh / 2],
            [xc + bw / 2, yc - bh / 2],
            [xc + bw / 2, yc + bh / 2],
            [xc - bw / 2, yc + bh / 2],
        ])

        # Rotate each corner around the image centre
        rotated = np.zeros_like(corners)
        for i, (x, y) in enumerate(corners):
            x -= cx;  y -= cy
            rotated[i] = [x * cos_ - y * sin_ + cx,
                          x * sin_ + y * cos_ + cy]

        # Axis-aligned bounding box of rotated corners
        x_min, y_min = rotated.min(axis=0)
        x_max, y_max = rotated.max(axis=0)

        # Clip to image bounds and renormalise
        x_min = np.clip(x_min, 0, img_w)
        y_min = np.clip(y_min, 0, img_h)
        x_max = np.clip(x_max, 0, img_w)
        y_max = np.clip(y_max, 0, img_h)

        new_xc = ((x_min + x_max) / 2) / img_w
        new_yc = ((y_min + y_max) / 2) / img_h
        new_w  = (x_max - x_min) / img_w
        new_h  = (y_max - y_min) / img_h

        if new_w > 0 and new_h > 0:
            new_labels.append((cls, new_xc, new_yc, new_w, new_h))

    return new_labels


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

def aug_brightness(img: np.ndarray) -> np.ndarray:
    """Brightness + contrast jitter — noticeably lighter or darker."""
    # Pick alpha clearly away from 1.0, beta clearly away from 0
    alpha = np.random.choice([0.6, 1.5])           # dark or bright
    beta  = int(np.random.choice([-50, 50]))
    return np.clip(img.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)


def aug_hsv(img: np.ndarray) -> np.ndarray:
    """Hue ±20, saturation ±60, value ±40 — clearly visible colour shift."""
    hsv          = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + np.random.randint(-20, 20), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(-60, 60), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.randint(-40, 40), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_blur(img: np.ndarray) -> np.ndarray:
    """Heavy Gaussian blur — clearly softer/out-of-focus look."""
    return cv2.GaussianBlur(img, (31, 31), 0)


def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w   = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# =============================================================================
# MAIN
# =============================================================================

def augment_dataset() -> None:
    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(image_files)} images — augmenting...")
    print(f"Output train: {os.path.abspath(OUT_TRAIN_IMAGES)}")
    print(f"Output val:   {os.path.abspath(OUT_VAL_IMAGES)}")

    for filename in image_files:
        stem       = os.path.splitext(filename)[0]
        img_path   = os.path.join(IMAGES_DIR, filename)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        img    = cv2.imread(img_path)
        labels = read_labels(label_path)

        if img is None:
            print(f"  [SKIP] Could not read {filename}")
            continue

        h, w = img.shape[:2]

        # ── Copy original → val ────────────────────────────────────────────
        save(stem, img, labels, train=False)

        # ── Flips → train ──────────────────────────────────────────────────
        if AUGMENTATIONS["flip_h"]:
            save(f"{stem}_fliph",
                 cv2.flip(img, 1), flip_h(labels), train=True)

        if AUGMENTATIONS["flip_v"]:
            save(f"{stem}_flipv",
                 cv2.flip(img, 0), flip_v(labels), train=True)

        # ── Rotations → train ─────────────────────────────────────────────
        for key, angle in [("rot_15", 15), ("rot_30", 30),
                            ("rot_neg15", -15), ("rot_neg30", -30)]:
            if AUGMENTATIONS[key]:
                save(f"{stem}_{key}",
                     rotate_image(img, angle),
                     rotate_labels(labels, angle, w, h), train=True)

        # ── Photometric → train ───────────────────────────────────────────
        if AUGMENTATIONS["brightness"]:
            save(f"{stem}_bright", aug_brightness(img), labels, train=True)

        if AUGMENTATIONS["hsv_shift"]:
            save(f"{stem}_hsv", aug_hsv(img), labels, train=True)

        if AUGMENTATIONS["blur"]:
            save(f"{stem}_blur", aug_blur(img), labels, train=True)

    train_total = len(os.listdir(OUT_TRAIN_IMAGES))
    val_total   = len(os.listdir(OUT_VAL_IMAGES))
    print(f"Done! {len(image_files)} originals → {train_total} train / {val_total} val")


if __name__ == "__main__":
    augment_dataset()
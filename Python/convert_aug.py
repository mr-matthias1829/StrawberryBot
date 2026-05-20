import cv2
import numpy as np
import os
import shutil

# this script is used to turn the images into training and validation data
# additionally, it makes variations of all the images. see config below for options and toggles
# to run: run this exact file. no other python files needed.
# note: make sure you ran convert_csv.py, as that creates the proper data that this script uses and needs

# =============================================================================
# CONFIG
# =============================================================================
IMAGES_DIR  = "dataset/images"
LABELS_DIR  = "dataset/labels"
OUT_TRAIN_IMAGES = "dataset_aug/images/train"
OUT_TRAIN_LABELS = "dataset_aug/labels/train"
OUT_VAL_IMAGES   = "dataset_aug/images/val"
OUT_VAL_LABELS   = "dataset_aug/labels/val"

SYNTHETIC_NEGATIVES_TRAIN = 30
SYNTHETIC_NEGATIVES_VAL   = 5

# Background-swap: how many random backgrounds to composite per image
BG_SWAP_TRAIN = 3   # dark / gradient / noisy variants each
BG_SWAP_VAL   = 1   # one random bg for val

AUGMENTATIONS = {
    "flip_h":     True,
    "flip_v":     True,
    "rot_15":     False,
    "rot_30":     False,
    "rot_neg15":  False,
    "rot_neg30":  False,
    "brightness": True,
    "hsv_shift":  True,
    "blur":       False,
    "zoom_in":    True,
    "zoom_out":   True,
    "zoom_out2":  False,
    "darken":     True,
    "heavy_blur": True,
    "bg_swap":    False,
}

# =============================================================================
# SETUP
# =============================================================================
for _dir in (OUT_TRAIN_IMAGES, OUT_TRAIN_LABELS, OUT_VAL_IMAGES, OUT_VAL_LABELS):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)


# =============================================================================
# LABEL HELPERS
# =============================================================================

def read_labels(label_path: str) -> list[tuple]:
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
    ok = cv2.imwrite(os.path.join(img_dir,   name + ".jpg"), img)
    if not ok:
        print(f"  [ERROR] Failed to write image: {name}.jpg")
        return
    write_labels(os.path.join(label_dir, name + ".txt"), labels)


# =============================================================================
# BBOX TRANSFORMS
# =============================================================================

def flip_h(labels):
    return [(c, 1.0-xc, yc, w, h) for c, xc, yc, w, h in labels]

def flip_v(labels):
    return [(c, xc, 1.0-yc, w, h) for c, xc, yc, w, h in labels]

def rotate_labels(labels, angle_deg, img_w, img_h):
    cx, cy     = img_w/2, img_h/2
    theta      = np.radians(-angle_deg)
    cos_, sin_ = np.cos(theta), np.sin(theta)
    new_labels = []
    for cls, xc_n, yc_n, w_n, h_n in labels:
        xc = xc_n*img_w; yc = yc_n*img_h
        bw = w_n*img_w;  bh = h_n*img_h
        corners = np.array([
            [xc-bw/2, yc-bh/2],[xc+bw/2, yc-bh/2],
            [xc+bw/2, yc+bh/2],[xc-bw/2, yc+bh/2],
        ])
        rotated = np.zeros_like(corners)
        for i, (x, y) in enumerate(corners):
            x -= cx; y -= cy
            rotated[i] = [x*cos_ - y*sin_ + cx, x*sin_ + y*cos_ + cy]
        x_min, y_min = rotated.min(axis=0)
        x_max, y_max = rotated.max(axis=0)
        x_min = np.clip(x_min, 0, img_w); y_min = np.clip(y_min, 0, img_h)
        x_max = np.clip(x_max, 0, img_w); y_max = np.clip(y_max, 0, img_h)
        nw = (x_max-x_min)/img_w; nh = (y_max-y_min)/img_h
        if nw > 0 and nh > 0:
            new_labels.append((cls, ((x_min+x_max)/2)/img_w,
                                    ((y_min+y_max)/2)/img_h, nw, nh))
    return new_labels

def zoom_in_labels(labels, crop_frac):
    offset = (1.0 - crop_frac) / 2.0
    new_labels = []
    for cls, xc, yc, w, h in labels:
        x1, y1 = xc-w/2, yc-h/2
        x2, y2 = xc+w/2, yc+h/2
        x1c = np.clip(x1, offset, offset+crop_frac)
        y1c = np.clip(y1, offset, offset+crop_frac)
        x2c = np.clip(x2, offset, offset+crop_frac)
        y2c = np.clip(y2, offset, offset+crop_frac)
        if w*h > 0 and (x2c-x1c)*(y2c-y1c)/(w*h) < 0.10:
            continue
        new_labels.append((cls,
            ((x1c+x2c)/2 - offset)/crop_frac,
            ((y1c+y2c)/2 - offset)/crop_frac,
            (x2c-x1c)/crop_frac,
            (y2c-y1c)/crop_frac))
    return new_labels

def zoom_out_labels(labels, pad_frac):
    scale  = 1.0 / (1.0 + pad_frac)
    offset = (1.0 - scale) / 2.0
    return [(cls, offset+xc*scale, offset+yc*scale, w*scale, h*scale)
            for cls, xc, yc, w, h in labels]


# =============================================================================
# BACKGROUND GENERATORS
# =============================================================================

def _dark_bg(h, w, rng):
    """Near-black — mimics dark surfaces, clothing."""
    v = int(rng.uniform(10, 55))
    bg = np.full((h, w, 3), v, dtype=np.uint8)
    # slight noise so it's not a flat block
    noise = rng.integers(-10, 11, (h, w, 3), dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def _gradient_bg(h, w, rng):
    """Two-color gradient — generic indoor/outdoor surface."""
    c1 = rng.integers(20, 210, 3).astype(np.float32)
    c2 = rng.integers(20, 210, 3).astype(np.float32)
    # random direction: horizontal or vertical
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    if rng.random() > 0.5:
        for i in range(h):
            bg[i, :] = (c1*(1-i/h) + c2*(i/h)).astype(np.uint8)
    else:
        for j in range(w):
            bg[:, j] = (c1*(1-j/w) + c2*(j/w)).astype(np.uint8)
    return bg

def _noisy_bg(h, w, rng):
    """Random noise — busy/textured surface."""
    base  = int(rng.uniform(30, 180))
    noise = rng.integers(-50, 51, (h, w, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def _wood_bg(h, w, rng):
    """Rough wood-grain simulation — common table surface."""
    base_color = np.array(rng.integers([60,30,10],[160,100,60], 3), dtype=np.float32)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        t = i / h
        bg[i, :] = base_color * (0.85 + 0.15 * t)
    # add horizontal grain lines
    n_lines = int(rng.uniform(8, 20))
    for _ in range(n_lines):
        y     = int(rng.uniform(0, h))
        thick = int(rng.uniform(1, 4))
        delta = int(rng.uniform(-20, 20))
        y1, y2 = max(0, y-thick), min(h, y+thick)
        bg[y1:y2, :] += delta
    noise = rng.uniform(-15, 15, (h, w, 3))
    return np.clip(bg + noise, 0, 255).astype(np.uint8)

def _solid_color_bg(h, w, rng):
    """Solid muted color — like a colored table or backdrop."""
    color = rng.integers(30, 200, 3)
    bg = np.full((h, w, 3), color, dtype=np.uint8)
    noise = rng.integers(-15, 16, (h, w, 3), dtype=np.int16)
    return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

_BG_FNS = [_dark_bg, _gradient_bg, _noisy_bg, _wood_bg, _solid_color_bg]

def random_background(h, w, rng=None, exclude=None):
    """Return a random background image of shape (h, w, 3)."""
    if rng is None:
        rng = np.random.default_rng()
    choices = [f for f in _BG_FNS if f is not exclude]
    fn = choices[rng.integers(len(choices))]
    return fn(h, w, rng)


# =============================================================================
# WHITE BACKGROUND REMOVAL + COMPOSITING
# =============================================================================

def extract_foreground_mask(img: np.ndarray,
                             white_thresh: int = 230,
                             erode_px: int = 2) -> np.ndarray:
    """
    Returns a uint8 mask (0=background, 255=foreground) by thresholding
    near-white pixels.  Works well for studio shots on white/light-grey bg.

    white_thresh : pixels where ALL channels >= this are considered background.
    erode_px     : shrinks the mask edge slightly to remove white fringing
                   that gets left behind after compositing.
    """
    # Convert to float and check if pixel is "near white" in all channels
    gray_max = img.max(axis=2)           # max channel per pixel
    gray_min = img.min(axis=2)           # min channel per pixel

    # Background: bright (all channels high) AND low saturation (max≈min)
    is_bg = (gray_max.astype(np.int16) >= white_thresh) & \
            (gray_max.astype(np.int16) - gray_min.astype(np.int16) < 30)

    mask = np.where(is_bg, 0, 255).astype(np.uint8)

    # Fill small holes inside the foreground (e.g. white glare on berries)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Erode to remove white halo fringing at the foreground edge
    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        mask = cv2.erode(mask, kernel_erode, iterations=1)

    return mask


def composite_on_background(img: np.ndarray,
                              mask: np.ndarray,
                              bg: np.ndarray) -> np.ndarray:
    """
    Alpha-blend foreground (img) over bg using mask.
    mask is uint8 0/255; we blur its edge for a smooth transition.
    """
    # Soft edge: blur the mask so the composite doesn't look cut-out
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]   # (H, W, 1)

    fg = img.astype(np.float32)
    background = bg.astype(np.float32)

    composite = fg * alpha + background * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def aug_bg_swap(img: np.ndarray, rng=None) -> np.ndarray:
    """Extract foreground, place on a random non-white background."""
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    mask = extract_foreground_mask(img)
    bg   = random_background(h, w, rng)
    return composite_on_background(img, mask, bg)


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

def aug_brightness(img):
    alpha = np.random.choice([0.6, 1.5])
    beta  = int(np.random.choice([-50, 50]))
    return np.clip(img.astype(np.int16)*alpha + beta, 0, 255).astype(np.uint8)

def aug_darken(img):
    return np.clip(img.astype(np.int16)*0.55 - 20, 0, 255).astype(np.uint8)

def aug_hsv(img):
    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + np.random.randint(-20, 20), 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] + np.random.randint(-60, 60), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + np.random.randint(-40, 40), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_blur(img):
    return cv2.GaussianBlur(img, (31, 31), 0)

def aug_heavy_blur(img):
    return cv2.GaussianBlur(img, (51, 51), 0)

def rotate_image(img, angle_deg):
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def aug_zoom_in(img, crop_frac=0.5):
    h, w    = img.shape[:2]
    pad_y   = int(h*(1.0-crop_frac)/2)
    pad_x   = int(w*(1.0-crop_frac)/2)
    cropped = img[pad_y:h-pad_y, pad_x:w-pad_x]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def aug_zoom_out(img, pad_frac=0.5):
    h, w      = img.shape[:2]
    new_h     = int(h/(1.0+pad_frac))
    new_w     = int(w/(1.0+pad_frac))
    resized   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top   = (h-new_h)//2
    pad_bot   = h-new_h-pad_top
    pad_left  = (w-new_w)//2
    pad_right = w-new_w-pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right,
                              cv2.BORDER_REPLICATE)


# =============================================================================
# SYNTHETIC NEGATIVE GENERATOR
# =============================================================================

_SKIN  = [(147,182,219),(120,160,200),(90,130,170),(60,90,130),(40,60,90),
           (130,160,210),(80,100,180),(100,130,200),(60,80,150),(150,180,230)]
_REDS  = [(30,30,180),(20,20,210),(40,50,200),(10,10,160),(50,60,190),
           (30,100,200),(20,80,210)]
_DARKS = [(20,20,20),(30,30,30),(15,15,15),(40,40,40),
           (10,20,30),(20,10,10),(25,25,35)]
_OTHER = [(60,160,60),(40,130,40),(180,180,180),(140,140,140),(30,100,200)]
_ALL_COLORS = _SKIN + _REDS + _DARKS + _OTHER


def _draw_blob(canvas, rng):
    size  = canvas.shape[0]
    color = tuple(int(v) for v in _ALL_COLORS[rng.integers(len(_ALL_COLORS))])
    shape = rng.choice(["round", "oval", "cylinder"])
    rx    = int(rng.uniform(0.06, 0.28) * size / 2)
    if shape == "round":
        ry = int(rng.uniform(0.85, 1.15) * rx)
    elif shape == "oval":
        ry = int(rng.uniform(0.5, 0.75) * rx)
    else:
        ry = int(rng.uniform(1.5, 2.5) * rx)
    margin = max(rx, ry) + 5
    if margin*2 >= size:
        margin = size//4
    cx    = int(rng.uniform(margin, size-margin))
    cy    = int(rng.uniform(margin, size-margin))
    angle = int(rng.uniform(0, 180))
    shadow = canvas.copy()
    cv2.ellipse(shadow, (cx, min(size-1, cy+ry+4)),
                (max(1,int(rx*1.2)), max(1,int(ry*0.3))),
                0, 0, 360, (180,180,180), -1)
    cv2.addWeighted(shadow, 0.15, canvas, 0.85, 0, canvas)
    for shrink, alpha in [(1.0,0.9),(0.72,0.6),(0.42,0.35)]:
        layer = canvas.copy()
        axes  = (max(1,int(rx*shrink)), max(1,int(ry*shrink)))
        bright = tuple(min(255, int(c+25*(1-shrink))) for c in color)
        cv2.ellipse(layer, (cx,cy), axes, angle, 0, 360, bright, -1)
        cv2.addWeighted(layer, alpha, canvas, 1-alpha, 0, canvas)


def generate_synthetic_negative(img_size: int = 640) -> np.ndarray:
    rng    = np.random.default_rng()
    # White bg weighted 2x so distribution matches real dataset
    bg_fn  = rng.choice([*_BG_FNS, _BG_FNS[0]])  # _BG_FNS[0] = dark, extra white via None
    # Actually: pick white half the time, random other half
    if rng.random() < 0.5:
        canvas = np.full((img_size, img_size, 3), 250, dtype=np.uint8)
    else:
        canvas = random_background(img_size, img_size, rng)
    for _ in range(int(rng.integers(1, 5))):
        _draw_blob(canvas, rng)
    noise  = rng.integers(-8, 9, canvas.shape, dtype=np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_all_negatives(n_train, n_val, img_size=640):
    print(f"Generating {n_train} train + {n_val} val synthetic negatives...")
    for i in range(n_train):
        save(f"synthetic_neg_{i:04d}",
             generate_synthetic_negative(img_size), [], train=True)
    for i in range(n_val):
        save(f"synthetic_neg_val_{i:04d}",
             generate_synthetic_negative(img_size), [], train=False)
    print(f"  Done — {n_train + n_val} negatives written.")


# =============================================================================
# MAIN
# =============================================================================

def augment_dataset() -> None:
    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(image_files)} images - augmenting...")
    print(f"Output train: {os.path.abspath(OUT_TRAIN_IMAGES)}")
    print(f"Output val:   {os.path.abspath(OUT_VAL_IMAGES)}")

    ZOOM_IN_FRAC   = 0.5
    ZOOM_OUT_FRAC  = 0.35
    ZOOM_OUT2_FRAC = 0.75

    rng = np.random.default_rng()

    # Fixed background sequence for train bg_swap so we always get one of each type
    _TRAIN_BG_FNS = [_dark_bg, _gradient_bg, _noisy_bg]  # length == BG_SWAP_TRAIN=3

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
        mask = extract_foreground_mask(img) if AUGMENTATIONS.get("bg_swap") else None

        # -- Copy original -> val ------------------------------------------
        save(stem, img, labels, train=False)

        # -- Flips -> train ------------------------------------------------
        if AUGMENTATIONS["flip_h"]:
            save(f"{stem}_fliph", cv2.flip(img,1), flip_h(labels), train=True)
        if AUGMENTATIONS["flip_v"]:
            save(f"{stem}_flipv", cv2.flip(img,0), flip_v(labels), train=True)

        # -- Rotations -> train --------------------------------------------
        for key, angle in [("rot_15",15),("rot_30",30),
                            ("rot_neg15",-15),("rot_neg30",-30)]:
            if AUGMENTATIONS[key]:
                save(f"{stem}_{key}",
                     rotate_image(img, angle),
                     rotate_labels(labels, angle, w, h), train=True)

        # -- Photometric -> train ------------------------------------------
        if AUGMENTATIONS["brightness"]:
            save(f"{stem}_bright", aug_brightness(img), labels, train=True)
        if AUGMENTATIONS["darken"]:
            save(f"{stem}_dark", aug_darken(img), labels, train=True)
        if AUGMENTATIONS["hsv_shift"]:
            save(f"{stem}_hsv", aug_hsv(img), labels, train=True)
        if AUGMENTATIONS["blur"]:
            save(f"{stem}_blur", aug_blur(img), labels, train=True)
        if AUGMENTATIONS["heavy_blur"]:
            save(f"{stem}_heavyblur", aug_heavy_blur(img), labels, train=True)

        # -- Spatial zoom -> train -----------------------------------------
        if AUGMENTATIONS["zoom_in"]:
            save(f"{stem}_zoomin",
                 aug_zoom_in(img, ZOOM_IN_FRAC),
                 zoom_in_labels(labels, ZOOM_IN_FRAC), train=True)
        if AUGMENTATIONS["zoom_out"]:
            save(f"{stem}_zoomout",
                 aug_zoom_out(img, ZOOM_OUT_FRAC),
                 zoom_out_labels(labels, ZOOM_OUT_FRAC), train=True)
        if AUGMENTATIONS["zoom_out2"]:
            save(f"{stem}_zoomout2",
                 aug_zoom_out(img, ZOOM_OUT2_FRAC),
                 zoom_out_labels(labels, ZOOM_OUT2_FRAC), train=True)

        # -- Background swap -> train + val --------------------------------
        if AUGMENTATIONS.get("bg_swap") and mask is not None:
            # Train: one of each bg type for variety
            for i, bg_fn in enumerate(_TRAIN_BG_FNS):
                bg      = bg_fn(h, w, rng)
                swapped = composite_on_background(img, mask, bg)
                save(f"{stem}_bg{i}", swapped, labels, train=True)

            # Val: one random bg
            bg      = random_background(h, w, rng)
            swapped = composite_on_background(img, mask, bg)
            save(f"{stem}_bgval", swapped, labels, train=False)

    # -- Synthetic hard negatives ------------------------------------------
    first    = cv2.imread(os.path.join(IMAGES_DIR, image_files[0]))
    neg_size = first.shape[0] if first is not None else 640
    generate_all_negatives(SYNTHETIC_NEGATIVES_TRAIN, SYNTHETIC_NEGATIVES_VAL,
                           img_size=neg_size)

    train_total = len(os.listdir(OUT_TRAIN_IMAGES))
    val_total   = len(os.listdir(OUT_VAL_IMAGES))
    print(f"Done! {len(image_files)} originals -> {train_total} train / {val_total} val")


if __name__ == "__main__":
    augment_dataset()
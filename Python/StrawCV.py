import cv2
import numpy as np
import os
from ultralytics import YOLO

# =============================================================================
# AI MODEL
# =============================================================================
model = YOLO("yolov8n.pt")


# =============================================================================
# CV PIPELINE — HSV + WATERSHED (file 1)
# =============================================================================

# ── HSV colour range for red (hue wraps around 0°) ───────────────────────────
RED_LOWER1, RED_UPPER1 = np.array([0,   100, 135]),  np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 100, 135]),  np.array([179, 255, 255])

# ── Morphology ────────────────────────────────────────────────────────────────
MORPH_OPEN_ITER  = 3   # noise removal passes
MORPH_CLOSE_ITER = 5   # gap-filling passes

# ── Primary seed detection ────────────────────────────────────────────────────
PRIMARY_PEAK_KERNEL    = 50  # minimum pixel distance between adjacent peaks
PRIMARY_PEAK_THRESHOLD = 12  # ignore peaks too close to the mask edge

# ── Colour-transition splitting ───────────────────────────────────────────────
TRANSITION_PERCENTILE     = 80.0
TRANSITION_PEAK_KERNEL    = 250
TRANSITION_PEAK_THRESHOLD = 15

# ── Fallback for large blobs that still have only one seed ───────────────────
FALLBACK_BLOB_MIN_AREA  = 10_000  # px² — blobs smaller than this are left alone
FALLBACK_PEAK_KERNEL    = 11
FALLBACK_PEAK_THRESHOLD = 5

# ── Final seed dilation before watershed ─────────────────────────────────────
SEED_DILATION = 25

# ── Post-processing: duplicate box merging ────────────────────────────────────
# Two boxes are merged when (intersection area / area of the smaller box)
# exceeds this ratio. Scale-independent: works the same at any zoom level.
# Raise toward 1.0 to only merge near-identical boxes; lower toward 0.5 if
# watershed noise still produces obvious duplicates. Avoid going below ~0.5
# or genuinely adjacent berries with touching boxes may get merged.
MERGE_OVERLAP_RATIO = 0.65


def build_red_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hsv, mask) where mask is a cleaned binary mask of red regions."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=MORPH_OPEN_ITER)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)
    return hsv, mask


def _local_maxima(dist: np.ndarray, kernel_size: int, threshold: float) -> np.ndarray:
    """Return a binary image with local maxima of *dist* above *threshold*."""
    local_max = cv2.dilate(dist, np.ones((kernel_size,) * 2, np.uint8))
    return np.uint8((dist == local_max) & (dist > threshold))


def _find_transition_lines(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a binary mask of likely berry-boundary lines inside *mask*."""
    sat_blur = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    val_blur = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)

    grad_sx = cv2.Sobel(sat_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_sy = cv2.Sobel(sat_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_vx = cv2.Sobel(val_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_vy = cv2.Sobel(val_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_sx + grad_vx, grad_sy + grad_vy)

    mask_pixels = grad_mag[mask > 0]
    if mask_pixels.size == 0:
        return np.zeros_like(mask)

    cutoff     = np.percentile(mask_pixels, TRANSITION_PERCENTILE)
    transition = np.uint8(grad_mag >= cutoff) * 255
    transition = cv2.bitwise_and(transition, mask)
    transition = cv2.dilate(transition, np.ones((3, 3), np.uint8), iterations=1)
    return transition


def _find_fallback_seeds(
    dist: np.ndarray,
    mask: np.ndarray,
    sure_fg: np.ndarray,
) -> np.ndarray:
    """Add seeds to large blobs that ended up with zero or one seed."""
    extra_peaks = _local_maxima(dist, FALLBACK_PEAK_KERNEL, FALLBACK_PEAK_THRESHOLD)

    num_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(mask)
    result = sure_fg.copy()

    for blob_id in range(1, num_blobs):
        if blob_stats[blob_id, cv2.CC_STAT_AREA] < FALLBACK_BLOB_MIN_AREA:
            continue

        blob_mask             = blob_labels == blob_id
        seeds_in_blob         = np.uint8(blob_mask & (sure_fg > 0))
        n_seed_components, _  = cv2.connectedComponents(seeds_in_blob)

        if n_seed_components <= 2:
            result[blob_mask & (extra_peaks > 0)] = 1

    return result


def find_seeds(
    dist: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sure_fg, unknown) seed maps for watershed initialisation."""
    # Stage 1: primary seeds
    sure_fg = _local_maxima(dist, PRIMARY_PEAK_KERNEL, PRIMARY_PEAK_THRESHOLD)

    # Stage 2: transition-aware seeds
    transition_lines = _find_transition_lines(hsv, mask)
    split_mask       = cv2.bitwise_and(mask, cv2.bitwise_not(transition_lines))
    dist_split       = cv2.distanceTransform(split_mask, cv2.DIST_L2, 5)
    transition_seeds = _local_maxima(dist_split, TRANSITION_PEAK_KERNEL, TRANSITION_PEAK_THRESHOLD)
    sure_fg          = cv2.bitwise_or(sure_fg, transition_seeds)

    # Stage 3: fallback for large blobs still lacking a second seed
    sure_fg = _find_fallback_seeds(dist, mask, sure_fg)

    # Dilate all seeds into stable watershed regions
    sure_fg = cv2.dilate(sure_fg, np.ones((SEED_DILATION,) * 2, np.uint8))
    unknown = cv2.subtract(mask, sure_fg)
    return sure_fg, unknown


def run_watershed(
    frame: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment *mask* into individual objects; return (dist_transform, markers)."""
    dist             = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg, unknown = find_seeds(dist, hsv, mask)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)

    return dist, markers


def depth_ordered_labels(markers: np.ndarray) -> dict[int, int]:
    """Map each watershed label to a 1-based depth order (lower Y = closer)."""
    object_labels = [l for l in np.unique(markers) if l > 1]
    center_y      = {l: np.mean(np.where(markers == l)[0]) for l in object_labels}
    ranked        = sorted(object_labels, key=lambda l: center_y[l])
    return {label: rank + 1 for rank, label in enumerate(ranked)}


def _draw_cv_results(
    frame: np.ndarray,
    markers: np.ndarray,
    label_order: dict[int, int],
) -> tuple[np.ndarray, int]:
    """Draw bounding boxes and depth-order numbers on each detected berry.

    Boxes are collected from all watershed regions first, then merged if their
    overlap ratio exceeds MERGE_OVERLAP_RATIO, and finally drawn. This collapses
    duplicate detections of the same berry caused by watershed noise without
    affecting well-separated berries whose boxes merely touch.
    """
    output = frame.copy()

    # Collect one box per valid contour across all labels
    raw_boxes = []
    for label in label_order:
        obj_mask    = np.uint8(markers == label)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1:
                continue
            raw_boxes.append(cv2.boundingRect(cnt))

    final_boxes = _merge_boxes(raw_boxes)

    for i, (x, y, w, h) in enumerate(final_boxes, start=1):
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output, str(i), (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output, len(final_boxes)


def _box_overlap_ratio(a: tuple, b: tuple) -> float:
    """Return intersection area / area of the smaller box.

    Boxes are (x, y, w, h). A ratio near 1.0 means one box is almost entirely
    inside the other — very likely the same berry detected twice.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0

    return intersection / min(aw * ah, bw * bh)


def _merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """Iteratively merge boxes whose overlap ratio exceeds MERGE_OVERLAP_RATIO.

    Merging produces the union bounding box. Repeats until stable so that
    chains of overlapping boxes collapse fully in one call.
    """
    merged = True
    while merged:
        merged   = False
        kept     = []
        absorbed = [False] * len(boxes)

        for i, box_a in enumerate(boxes):
            if absorbed[i]:
                continue
            current = box_a
            for j, box_b in enumerate(boxes):
                if i == j or absorbed[j]:
                    continue
                if _box_overlap_ratio(current, box_b) >= MERGE_OVERLAP_RATIO:
                    ax1, ay1, aw, ah = current
                    bx1, by1, bw, bh = box_b
                    nx1 = min(ax1, bx1)
                    ny1 = min(ay1, by1)
                    nx2 = max(ax1 + aw, bx1 + bw)
                    ny2 = max(ay1 + ah, by1 + bh)
                    current     = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    absorbed[j] = True
                    merged      = True
            kept.append(current)
        boxes = kept

    return boxes


def _dist_to_view(dist: np.ndarray) -> np.ndarray:
    """Normalise a distance transform to a displayable uint8 image."""
    return cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def cv_pipeline(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Full CV pipeline: HSV masking → watershed → bounding boxes.

    Returns (annotated_frame, mask, dist_view, berry_count).
    """
    hsv, mask     = build_red_mask(frame)
    dist, markers = run_watershed(frame, hsv, mask)
    label_order   = depth_ordered_labels(markers)
    output, count = _draw_cv_results(frame, markers, label_order)
    return output, mask, _dist_to_view(dist), count


# =============================================================================
# AI PIPELINE — YOLO (file 2)
# =============================================================================

def ai_pipeline(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Run YOLOv8 inference; return (annotated_frame, detection_count)."""
    results = model(frame)
    output  = frame.copy()
    count   = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

    return output, count


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_webcam() -> None:
    """Live webcam loop: runs both pipelines side-by-side."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam niet gevonden!")
        return

    print("Webcam gestart (druk 'q' om te stoppen)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv_out, mask, dist_view, cv_count = cv_pipeline(frame)
        ai_out, ai_count                  = ai_pipeline(frame)

        combined = frame.copy()
        cv2.putText(combined, f"CV: {cv_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined, f"AI: {ai_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Combined", combined)
        cv2.imshow("AI",       ai_out)
        cv2.imshow("CV",       cv_out)
        cv2.imshow("Mask",     mask)
        cv2.imshow("Distance", dist_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image() -> None:
    """Single-image mode: load from Assets folder."""
    img_path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")
    frame    = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    cv_out, mask, dist_view, count = cv_pipeline(frame)

    cv2.imshow("Distance", dist_view)
    cv2.imshow("Result",   cv_out)
    print(f"Detected strawberries: {count}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change to run_image() to use single-image mode.
    run_webcam()
# =============================================================================
# detection.py — detection engine
#
# Three public functions:
#   cv_pipeline(frame)           → (annotated, mask, dist_view, boxes)
#   ai_pipeline(frame, conf)     → (annotated, boxes)
#   cv_score_crop(frame, box)    → float 0..1
#
# "boxes" are always lists of (x1, y1, x2, y2) ints so fusion can IoU-match
# them without knowing which detector produced them.
# =============================================================================

import cv2
import numpy as np
from ultralytics import YOLO

import config as cfg

# ── Load model once at import time ────────────────────────────────────────────
_model = YOLO(cfg.MODEL_PATH)


# =============================================================================
# CV PIPELINE — HSV + Watershed
# =============================================================================

def _adaptive_peak_kernel(area: float) -> int:
    radius = np.sqrt(area / np.pi)
    return max(cfg.PRIMARY_PEAK_KERNEL, int(radius * cfg.ADAPTIVE_SCALE))


def _adaptive_close(mask: np.ndarray) -> np.ndarray:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return mask
    areas    = stats[1:, cv2.CC_STAT_AREA]
    median_r = float(np.sqrt(np.median(areas) / np.pi))
    k        = int(np.clip(median_r * 0.20, 3, 31))
    k        = k if k % 2 == 1 else k + 1
    kernel   = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _build_red_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.bitwise_or(
        cv2.inRange(hsv, np.array(cfg.RED_LOWER1), np.array(cfg.RED_UPPER1)),
        cv2.inRange(hsv, np.array(cfg.RED_LOWER2), np.array(cfg.RED_UPPER2)),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=cfg.MORPH_OPEN_ITER)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.MORPH_CLOSE_ITER)
    mask   = _adaptive_close(mask)
    return hsv, mask


def _local_maxima(
    dist: np.ndarray,
    kernel_size: int,
    threshold: float,
    smooth_k: int = 15,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    dist_smooth = cv2.GaussianBlur(dist, (smooth_k, smooth_k), smooth_sigma)
    local_max   = cv2.dilate(dist_smooth, np.ones((kernel_size,) * 2, np.uint8))
    return np.uint8((dist_smooth == local_max) & (dist_smooth > threshold))


def _find_transition_lines(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sat_blur = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    val_blur = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    grad_sx  = cv2.Sobel(sat_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_sy  = cv2.Sobel(sat_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_vx  = cv2.Sobel(val_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_vy  = cv2.Sobel(val_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_sx + grad_vx, grad_sy + grad_vy)
    mask_pixels = grad_mag[mask > 0]
    if mask_pixels.size == 0:
        return np.zeros_like(mask)
    cutoff     = np.percentile(mask_pixels, cfg.TRANSITION_PERCENTILE)
    transition = np.uint8(grad_mag >= cutoff) * 255
    transition = cv2.bitwise_and(transition, mask)
    return cv2.dilate(transition, np.ones((3, 3), np.uint8), iterations=1)


def _find_fallback_seeds(
    dist: np.ndarray,
    mask: np.ndarray,
    sure_fg: np.ndarray,
) -> np.ndarray:
    extra_peaks = _local_maxima(dist, cfg.FALLBACK_PEAK_KERNEL, cfg.FALLBACK_PEAK_THRESHOLD)
    num_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(mask)
    result = sure_fg.copy()
    for blob_id in range(1, num_blobs):
        if blob_stats[blob_id, cv2.CC_STAT_AREA] < cfg.FALLBACK_BLOB_MIN_AREA:
            continue
        blob_mask            = blob_labels == blob_id
        seeds_in_blob        = np.uint8(blob_mask & (sure_fg > 0))
        n_seed_components, _ = cv2.connectedComponents(seeds_in_blob)
        if n_seed_components < 2:
            result[blob_mask & (extra_peaks > 0)] = 1
    return result


def _find_seeds(
    dist: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, blob_stats, _ = cv2.connectedComponentsWithStats(mask)
    if blob_stats.shape[0] > 1:
        max_area        = float(blob_stats[1:, cv2.CC_STAT_AREA].max())
        adaptive_kernel = _adaptive_peak_kernel(max_area)
        radius   = np.sqrt(max_area / np.pi)
        smooth_k = int(np.clip(radius * 0.12, 15, 61))
        smooth_k = smooth_k if smooth_k % 2 == 1 else smooth_k + 1
        smooth_s = float(smooth_k * 0.25)
    else:
        adaptive_kernel    = cfg.PRIMARY_PEAK_KERNEL
        smooth_k, smooth_s = 15, 3.0

    sure_fg          = _local_maxima(dist, adaptive_kernel, cfg.PRIMARY_PEAK_THRESHOLD, smooth_k, smooth_s)
    transition_lines = _find_transition_lines(hsv, mask)
    split_mask       = cv2.bitwise_and(mask, cv2.bitwise_not(transition_lines))
    dist_split       = cv2.distanceTransform(split_mask, cv2.DIST_L2, 5)
    transition_seeds = _local_maxima(dist_split, cfg.TRANSITION_PEAK_KERNEL, cfg.TRANSITION_PEAK_THRESHOLD)
    sure_fg          = cv2.bitwise_or(sure_fg, transition_seeds)
    sure_fg          = _find_fallback_seeds(dist, mask, sure_fg)
    sure_fg          = cv2.dilate(sure_fg, np.ones((cfg.SEED_DILATION,) * 2, np.uint8))
    unknown          = cv2.subtract(mask, sure_fg)
    return sure_fg, unknown


def _run_watershed(
    frame: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dist             = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg, unknown = _find_seeds(dist, hsv, mask)
    _, markers       = cv2.connectedComponents(sure_fg)
    markers         += 1
    markers[unknown == 255] = 0
    markers          = cv2.watershed(frame, markers)
    return dist, markers


def _box_overlap_ratio(a: tuple, b: tuple) -> float:
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
                if _box_overlap_ratio(current, box_b) >= cfg.MERGE_OVERLAP_RATIO:
                    ax1, ay1, aw, ah = current
                    bx1, by1, bw, bh = box_b
                    nx1 = min(ax1, bx1);  ny1 = min(ay1, by1)
                    nx2 = max(ax1+aw, bx1+bw); ny2 = max(ay1+ah, by1+bh)
                    current     = (nx1, ny1, nx2-nx1, ny2-ny1)
                    absorbed[j] = True
                    merged      = True
            kept.append(current)
        boxes = kept
    return boxes


def _xywh_to_xyxy(boxes_xywh: list[tuple]) -> list[tuple]:
    return [(x, y, x+w, y+h) for x, y, w, h in boxes_xywh]


def cv_pipeline(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """HSV masking → watershed → bounding boxes.

    Returns (annotated_frame, mask, dist_view, boxes_xyxy).
    boxes_xyxy is a list of (x1, y1, x2, y2) for fusion.
    """
    hsv, mask     = _build_red_mask(frame)
    dist, markers = _run_watershed(frame, hsv, mask)

    # collect raw boxes from watershed labels
    raw_boxes = []
    for label in [l for l in np.unique(markers) if l > 1]:
        obj_mask    = np.uint8(markers == label)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1:
                continue
            raw_boxes.append(cv2.boundingRect(cnt))

    merged  = _merge_boxes(raw_boxes)
    final   = [(x, y, w, h) for x, y, w, h in merged if w * h >= cfg.MIN_BOX_AREA]
    boxes   = _xywh_to_xyxy(final)

    # draw
    output = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(output, str(i), (x1+4, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return output, mask, dist_view, boxes


# =============================================================================
# AI PIPELINE — YOLO
# =============================================================================

def ai_pipeline(frame: np.ndarray, conf: float | None = None) -> tuple[np.ndarray, list[tuple]]:
    """Run YOLO on frame.

    Returns (annotated_frame, boxes_xyxy).
    boxes_xyxy includes the confidence as a 5th element: (x1,y1,x2,y2,conf).
    """
    threshold = conf if conf is not None else cfg.YOLO_CONF_THRESHOLD
    results   = _model(frame, conf=threshold)
    output    = frame.copy()
    boxes     = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score           = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, score))
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"{score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return output, boxes


# =============================================================================
# CV SCORER — sanity-check a single YOLO crop
# =============================================================================

def cv_score_crop(frame: np.ndarray, box: tuple) -> dict:
    """Score a single bounding box crop using CV features.

    Args:
        frame: full BGR frame
        box:   (x1, y1, x2, y2) or (x1, y1, x2, y2, conf)

    Returns a dict with individual scores and the weighted total (0..1).
    Useful for the debug logger.
    """
    x1, y1, x2, y2 = box[:4]
    # clamp to frame bounds
    h_fr, w_fr = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_fr, x2), min(h_fr, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"redness": 0, "circularity": 0, "size": 0, "texture": 0, "total": 0.0}

    # ── Redness ───────────────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array(cfg.RED_LOWER1), np.array(cfg.RED_UPPER1))
    red2 = cv2.inRange(hsv, np.array(cfg.RED_LOWER2), np.array(cfg.RED_UPPER2))
    red_pixels  = cv2.countNonZero(cv2.bitwise_or(red1, red2))
    total_pixels = crop.shape[0] * crop.shape[1]
    redness = min(1.0, red_pixels / max(total_pixels * 0.3, 1))

    # ── Circularity (contour of the red region) ───────────────────────────────
    red_mask    = cv2.bitwise_or(red1, red2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt  = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri > 0:
            circularity = min(1.0, (4 * np.pi * area) / (peri ** 2))

    # ── Size match ────────────────────────────────────────────────────────────
    box_area = (x2 - x1) * (y2 - y1)
    if box_area < cfg.BERRY_SIZE_MIN:
        size_score = box_area / cfg.BERRY_SIZE_MIN
    elif box_area > cfg.BERRY_SIZE_MAX:
        size_score = cfg.BERRY_SIZE_MAX / box_area
    else:
        # peak at BERRY_SIZE_IDEAL, linear ramp either side
        if box_area <= cfg.BERRY_SIZE_IDEAL:
            size_score = (box_area - cfg.BERRY_SIZE_MIN) / max(cfg.BERRY_SIZE_IDEAL - cfg.BERRY_SIZE_MIN, 1)
        else:
            size_score = (cfg.BERRY_SIZE_MAX - box_area) / max(cfg.BERRY_SIZE_MAX - cfg.BERRY_SIZE_IDEAL, 1)
        size_score = float(np.clip(size_score, 0.0, 1.0))

    # ── Texture (Laplacian variance — berries have surface detail) ────────────
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalise: a completely smooth patch ≈ 0, a typical berry ≈ 200-800
    texture = float(np.clip(lap_var / 500.0, 0.0, 1.0))

    # ── Weighted total ────────────────────────────────────────────────────────
    total = (
        cfg.CV_WEIGHT_REDNESS     * redness     +
        cfg.CV_WEIGHT_CIRCULARITY * circularity +
        cfg.CV_WEIGHT_SIZE        * size_score  +
        cfg.CV_WEIGHT_TEXTURE     * texture
        # temporal weight is applied by fusion after seeing seen_count
    )

    return {
        "redness":     round(redness,     3),
        "circularity": round(circularity, 3),
        "size":        round(size_score,  3),
        "texture":     round(texture,     3),
        "total":       round(total,       3),
    }
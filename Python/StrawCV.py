import cv2
import numpy as np
import os

# ── HSV colour range for red (hue wraps around 0°) ───────────────────────────
RED_LOWER1, RED_UPPER1 = np.array([0,   50, 50]), np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 50, 50]), np.array([179, 255, 255])

# ── Morphology ────────────────────────────────────────────────────────────────
MORPH_OPEN_ITER  = 3   # noise removal passes
MORPH_CLOSE_ITER = 5   # gap-filling passes

# ── Primary seed detection ────────────────────────────────────────────────────
PRIMARY_PEAK_KERNEL    = 50  # minimum pixel distance between adjacent peaks
PRIMARY_PEAK_THRESHOLD = 12   # ignore peaks too close to the mask edge

# ── Colour-transition splitting ───────────────────────────────────────────────
# Keep only the top N% of gradient magnitudes (within the red mask) as
# transition lines. Higher = more aggressive splitting; lower = more passive.
# 85 is a safe starting point; raise toward 95 if berries are still merging,
# lower toward 70 if a single berry is being cut in half.
TRANSITION_PERCENTILE  = 80.0
TRANSITION_PEAK_KERNEL = 250  # peak spacing after splitting on transition lines
TRANSITION_PEAK_THRESHOLD = 15

# ── Fallback for large blobs that still have only one seed ───────────────────
FALLBACK_BLOB_MIN_AREA   = 10_000  # px² — blobs smaller than this are left alone
FALLBACK_PEAK_KERNEL     = 11
FALLBACK_PEAK_THRESHOLD  = 5

# ── Final seed dilation before watershed ─────────────────────────────────────
SEED_DILATION = 25  # grow seeds into stable regions; smaller than single-berry
                    # version to avoid merging nearby seeds back together


# ─────────────────────────────────────────────────────────────────────────────

def build_red_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hsv, mask) where mask is a cleaned binary mask of red regions."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
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
    """Return a binary mask of likely berry-boundary lines inside *mask*.

    Combines saturation and value gradients, then keeps only the strongest
    TRANSITION_PERCENTILE percent — making aggressiveness a single, stable knob
    rather than an Otsu threshold that shifts unpredictably between images.
    If the gradient signal inside the mask is too weak (all berries well-separated),
    the function returns an empty mask so downstream logic isn't misled.
    """
    sat_blur = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    val_blur = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)

    grad_sx = cv2.Sobel(sat_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_sy = cv2.Sobel(sat_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_vx = cv2.Sobel(val_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_vy = cv2.Sobel(val_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_sx + grad_vx, grad_sy + grad_vy)

    # Threshold at percentile of gradient values *within the mask only*.
    # This is image-scale-independent: "top 15% inside the berry region"
    # means the same thing regardless of overall image contrast.
    mask_pixels = grad_mag[mask > 0]
    if mask_pixels.size == 0:
        return np.zeros_like(mask)

    cutoff = np.percentile(mask_pixels, TRANSITION_PERCENTILE)
    transition = np.uint8(grad_mag >= cutoff) * 255
    transition = cv2.bitwise_and(transition, mask)
    transition = cv2.dilate(transition, np.ones((3, 3), np.uint8), iterations=1)
    return transition


def _find_fallback_seeds(
    dist: np.ndarray,
    mask: np.ndarray,
    sure_fg: np.ndarray,
) -> np.ndarray:
    """Add seeds to large blobs that ended up with zero or one seed.

    This catches cases where both primary and transition detection missed an
    overlap — if a blob is big enough to be two berries but has only one seed,
    we force-add a secondary peak from a finer local-max search.
    """
    extra_peaks = _local_maxima(dist, FALLBACK_PEAK_KERNEL, FALLBACK_PEAK_THRESHOLD)

    num_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(mask)
    result = sure_fg.copy()

    for blob_id in range(1, num_blobs):
        if blob_stats[blob_id, cv2.CC_STAT_AREA] < FALLBACK_BLOB_MIN_AREA:
            continue  # small enough to be a single berry — leave it alone

        blob_mask    = blob_labels == blob_id
        seeds_in_blob = np.uint8(blob_mask & (sure_fg > 0))
        n_seed_components, _ = cv2.connectedComponents(seeds_in_blob)

        if n_seed_components <= 2:  # only background + (at most) one seed region
            result[blob_mask & (extra_peaks > 0)] = 1

    return result


def find_seeds(
    dist: np.ndarray,
    hsv: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sure_fg, unknown) seed maps for watershed initialisation.

    Three-stage strategy:
      1. Primary seeds   — local maxima on the full distance transform.
      2. Transition seeds — secondary maxima found after erasing likely
                           berry boundaries from the mask.
      3. Fallback seeds  — force-split any large blob still left with one seed.
    """
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
    dist            = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sure_fg, unknown = find_seeds(dist, hsv, mask)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)

    return dist, markers


def depth_ordered_labels(markers: np.ndarray) -> dict[int, int]:
    """Map each watershed label to a 1-based depth order.

    Lower Y coordinate → smaller order number (closer to camera).
    """
    object_labels = [l for l in np.unique(markers) if l > 1]
    center_y      = {l: np.mean(np.where(markers == l)[0]) for l in object_labels}
    ranked        = sorted(object_labels, key=lambda l: center_y[l])
    return {label: rank + 1 for rank, label in enumerate(ranked)}


def draw_results(
    frame: np.ndarray,
    markers: np.ndarray,
    label_order: dict[int, int],
) -> tuple[np.ndarray, int]:
    """Draw a bounding box and depth-order number on each detected berry.

    Returns (annotated image, berry count).
    """
    output = frame.copy()
    count  = 0

    for label, order in label_order.items():
        obj_mask    = np.uint8(markers == label)
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1:  # TODO: make threshold dynamic
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output, str(order), (x + 4, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            count += 1

    return output, count


def main() -> None:
    img_path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")
    frame    = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    hsv, mask           = build_red_mask(frame)
    dist, markers       = run_watershed(frame, hsv, mask)
    label_order         = depth_ordered_labels(markers)
    result, count       = draw_results(frame, markers, label_order)

    dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("Distance", dist_view)
    cv2.imshow("Result",   result)
    print(f"Detected strawberries: {count}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
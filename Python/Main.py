# =============================================================================
# main.py — orchestrator
#
# Wires cv_pipeline + ai_pipeline together through:
#   • IoU box matching
#   • Fusion / decision engine  (Cases A-D from the GPT spec)
#   • Temporal tracker          (NEW → VERIFY → CONFIRMED / REJECTED)
#   • Debug logger
#
# Entry points:
#   run_webcam(show_windows=True)
#   run_image(path=None)
# =============================================================================

import os
import cv2
import numpy as np

import config as cfg
from detection import cv_pipeline, ai_pipeline, cv_score_crop


# =============================================================================
# IoU helpers
# =============================================================================

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2[,...]) boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(union, 1)


def _match_boxes(ai_boxes, cv_boxes):
    """Pair each AI box with its best-matching CV box (if IoU ≥ threshold).

    Returns:
        matched   — list of (ai_box, cv_box) pairs
        ai_only   — AI boxes with no CV match
        cv_only   — CV boxes with no AI match
    """
    used_cv  = [False] * len(cv_boxes)
    matched  = []
    ai_only  = []

    for ai in ai_boxes:
        best_iou, best_j = 0.0, -1
        for j, cv in enumerate(cv_boxes):
            if used_cv[j]:
                continue
            score = _iou(ai, cv)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= cfg.IOU_MATCH_THRESHOLD:
            matched.append((ai, cv_boxes[best_j]))
            used_cv[best_j] = True
        else:
            ai_only.append(ai)

    cv_only = [cv for j, cv in enumerate(cv_boxes) if not used_cv[j]]
    return matched, ai_only, cv_only


# =============================================================================
# Temporal tracker
# =============================================================================

class _Track:
    __slots__ = ("box", "seen", "missed", "state", "rechecks")

    def __init__(self, box):
        self.box      = box          # (x1,y1,x2,y2)
        self.seen     = 1
        self.missed   = 0
        self.state    = "NEW"        # NEW | CONFIRMED | REJECTED
        self.rechecks = 0


class Tracker:
    def __init__(self):
        self._tracks: list[_Track] = []

    def update(self, confirmed_boxes: list[tuple]) -> list[tuple]:
        """Feed this frame's confirmed boxes; returns only CONFIRMED tracks' boxes."""
        used = [False] * len(confirmed_boxes)

        # link new boxes to existing tracks
        for t in self._tracks:
            best_iou, best_i = 0.0, -1
            for i, box in enumerate(confirmed_boxes):
                if used[i]:
                    continue
                score = _iou(t.box, box)
                if score > best_iou:
                    best_iou, best_i = score, i

            if best_iou >= cfg.TRACKER_IOU_LINK:
                t.box    = confirmed_boxes[best_i]
                t.seen  += 1
                t.missed = 0
                used[best_i] = True
                if t.seen >= cfg.TRACKER_CONFIRM_COUNT:
                    t.state = "CONFIRMED"
            else:
                t.missed += 1
                if t.missed >= cfg.TRACKER_MISS_COUNT:
                    t.state = "REJECTED"

        # new tracks for unmatched boxes
        for i, box in enumerate(confirmed_boxes):
            if not used[i]:
                self._tracks.append(_Track(box))

        # prune dead tracks
        self._tracks = [t for t in self._tracks if t.state != "REJECTED"]

        return [t.box for t in self._tracks if t.state == "CONFIRMED"]


# =============================================================================
# Debug logger
# =============================================================================

def _log(ai_conf: float, cv_scores: dict, fused: float, decision: str) -> None:
    print(
        f"  YOLO:{ai_conf:.2f}  "
        f"Red:{cv_scores.get('redness',0):.2f}  "
        f"Circ:{cv_scores.get('circularity',0):.2f}  "
        f"Size:{cv_scores.get('size',0):.2f}  "
        f"Tex:{cv_scores.get('texture',0):.2f}  "
        f"CV:{cv_scores.get('total',0):.2f}  "
        f"Fused:{fused:.2f}  → {decision}"
    )


# =============================================================================
# Fusion / decision engine
# =============================================================================

def _zoom_crop(frame: np.ndarray, box: tuple, scale: float = 2.0) -> np.ndarray:
    """Return a frame-sized image with the crop region upscaled (fills frame)."""
    x1, y1, x2, y2 = box[:4]
    crop  = frame[max(0,y1):min(frame.shape[0],y2),
                  max(0,x1):min(frame.shape[1],x2)]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, (frame.shape[1], frame.shape[0]))


def _fuse(ai_conf: float, cv_total: float) -> float:
    return cfg.FUSION_YOLO_WEIGHT * ai_conf + cfg.FUSION_CV_WEIGHT * cv_total


def run_fusion(
    frame: np.ndarray,
    ai_boxes: list[tuple],
    cv_boxes: list[tuple],
    tracker: Tracker,
    recheck_depth: int = 0,
) -> tuple[list[tuple], np.ndarray]:
    """Decision engine — Cases A/B/C/D.

    Returns (confirmed_boxes, annotated_frame).
    confirmed_boxes are (x1,y1,x2,y2) ready for the tracker.
    """
    matched, ai_only, cv_only = _match_boxes(ai_boxes, cv_boxes)
    candidates = []   # (box, ai_conf, cv_scores, decision_label)

    # ── Case A: both agree ────────────────────────────────────────────────────
    for ai_box, cv_box in matched:
        ai_conf  = ai_box[4] if len(ai_box) > 4 else 0.0
        cv_sc    = cv_score_crop(frame, ai_box)
        fused    = _fuse(ai_conf, cv_sc["total"])
        decision = "ACCEPT" if fused >= cfg.FUSION_THRESHOLD else "WEAK"
        _log(ai_conf, cv_sc, fused, f"Case A → {decision}")
        if fused >= cfg.FUSION_THRESHOLD:
            candidates.append(ai_box[:4])

    # ── Case B: AI yes, CV no ─────────────────────────────────────────────────
    for ai_box in ai_only:
        ai_conf = ai_box[4] if len(ai_box) > 4 else 0.0
        cv_sc   = cv_score_crop(frame, ai_box)
        fused   = _fuse(ai_conf, cv_sc["total"])

        if ai_conf >= cfg.YOLO_ZOOM_THRESHOLD:
            _log(ai_conf, cv_sc, fused, "Case B → ACCEPT (high AI conf)")
            candidates.append(ai_box[:4])
        elif recheck_depth < cfg.MAX_RECHECKS:
            _log(ai_conf, cv_sc, fused, f"Case B → ZOOM (depth {recheck_depth+1})")
            zoomed          = _zoom_crop(frame, ai_box)
            _, zoom_ai      = ai_pipeline(zoomed, conf=cfg.YOLO_CONF_THRESHOLD * 0.8)
            zoom_cv_out, _, _, zoom_cv = cv_pipeline(zoomed)
            zoom_candidates, _ = run_fusion(zoomed, zoom_ai, zoom_cv, tracker,
                                            recheck_depth + 1)
            # map zoom coords back — mark originals as accepted if zoom confirmed
            if zoom_candidates:
                candidates.append(ai_box[:4])
        else:
            _log(ai_conf, cv_sc, fused, "Case B → REJECT (max rechecks)")

    # ── Case C: CV yes, AI no ─────────────────────────────────────────────────
    for cv_box in cv_only:
        if recheck_depth < cfg.MAX_RECHECKS:
            _log(0.0, {}, 0.0, f"Case C → ZOOM AI (depth {recheck_depth+1})")
            zoomed     = _zoom_crop(frame, cv_box)
            _, zoom_ai = ai_pipeline(zoomed, conf=cfg.YOLO_CONF_THRESHOLD * 0.8)
            if zoom_ai:
                best_ai  = max(zoom_ai, key=lambda b: b[4])
                ai_conf  = best_ai[4]
                cv_sc    = cv_score_crop(frame, cv_box)
                fused    = _fuse(ai_conf, cv_sc["total"])
                _log(ai_conf, cv_sc, fused, f"Case C zoom → {'ACCEPT' if fused >= cfg.FUSION_THRESHOLD else 'REJECT'}")
                if fused >= cfg.FUSION_THRESHOLD:
                    candidates.append(cv_box[:4])
            else:
                _log(0.0, {}, 0.0, "Case C zoom → REJECT (AI still blind)")
        else:
            _log(0.0, {}, 0.0, "Case C → REJECT (max rechecks)")

    # ── Temporal confirmation ─────────────────────────────────────────────────
    confirmed = tracker.update(candidates)

    # ── Annotate ──────────────────────────────────────────────────────────────
    output = frame.copy()
    for x1, y1, x2, y2 in confirmed:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 50), 3)
        cv2.putText(output, "berry", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 50), 2)

    return confirmed, output


# =============================================================================
# Entry points
# =============================================================================

def run_webcam(show_windows: bool = True) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam niet gevonden!")
        return

    print("Webcam gestart (druk 'q' om te stoppen)")
    tracker = Tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv_out, mask, dist_view, cv_boxes = cv_pipeline(frame)
        ai_out, ai_boxes                  = ai_pipeline(frame)
        confirmed, fused_out              = run_fusion(frame, ai_boxes, cv_boxes, tracker)

        print(f"Bevestigde aardbeien: {len(confirmed)}")

        if show_windows:
            cv2.imshow("Fused",    fused_out)
            cv2.imshow("AI",       ai_out)
            cv2.imshow("CV",       cv_out)
            cv2.imshow("Mask",     mask)
            cv2.imshow("Distance", dist_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str | None = None) -> None:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")

    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {path}")

    tracker = Tracker()

    cv_out, mask, dist_view, cv_boxes = cv_pipeline(frame)
    ai_out, ai_boxes                  = ai_pipeline(frame)
    confirmed, fused_out              = run_fusion(frame, ai_boxes, cv_boxes, tracker)

    print(f"Bevestigde aardbeien: {len(confirmed)}")

    cv2.imshow("Fused",    fused_out)
    cv2.imshow("AI",       ai_out)
    cv2.imshow("CV",       cv_out)
    cv2.imshow("Mask",     mask)
    cv2.imshow("Distance", dist_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Switch to run_image() for still-image testing
    run_webcam(show_windows=True)
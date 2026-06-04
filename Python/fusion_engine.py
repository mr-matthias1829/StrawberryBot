import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from detection import AIDetector, CVDectector, Detection, CLASS_COLORS, CLASS_NAMES, iou
from robot_controller import RobotController


INFER_SCALE: float = 1          # inference resolution multiplier
DETECT_EVERY: int  = 1          # run detection every N frames
RECHECK_EVERY_N_DETECTIONS: int = 3
ZOOM_QUEUE_MAXSIZE: int         = 4
CLEANUP_INTERVAL: int           = 30

# Containment score supplement to standard IoU.
# Handles the case where a small CV box sits entirely inside a larger AI box.
CONTAINMENT_MATCH_THRESHOLD = 0.45

# ── Minimum box size for targeting ───────────────────────────────────────────
MIN_TARGET_BOX_AREA: int = 900   # pixels²

# ── Movement prediction ───────────────────────────────────────────────────────
# Smoothing factor for the exponential moving average velocity (0 = no update,
# 1 = instant update).  Lower values produce smoother but laggier predictions.
VELOCITY_ALPHA: float = 0.4

# Maximum age (seconds) of a tracker's last real detection before its predicted
# position is no longer drawn.  Keeps stale ghosts off the screen.
MAX_PREDICT_AGE: float = 0.25   # seconds

# How many frames back we look to compute the initial velocity estimate.
# Increasing this smooths out noise but makes the estimate slower to react.
VELOCITY_HISTORY_LEN: int = 6


# =============================================================================
# AI TOGGLE
# =============================================================================
_ai_enabled      = True
_ai_enabled_lock = threading.Lock()


def set_ai_enabled(enabled: bool) -> None:
    global _ai_enabled
    with _ai_enabled_lock:
        _ai_enabled = bool(enabled)
    print(f"[FusionEngine] AI detector {'ENABLED' if enabled else 'DISABLED'}")


def is_ai_enabled() -> bool:
    with _ai_enabled_lock:
        return _ai_enabled


# =============================================================================
# TRACKED OBJECT
# =============================================================================

@dataclass
class TrackedObject:
    id: int
    detection: Detection
    seen_count: int = 1
    missed_count: int = 0
    fused_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)

    # ── Velocity / prediction ─────────────────────────────────────────────────
    # Velocity is stored in pixels-per-second for both axes, as an EMA.
    _vx: float = field(default=0.0, init=False, repr=False)
    _vy: float = field(default=0.0, init=False, repr=False)
    _last_cx: float = field(default=0.0, init=False, repr=False)
    _last_cy: float = field(default=0.0, init=False, repr=False)
    _last_update_time: float = field(default_factory=time.time, init=False, repr=False)

    # Ring buffer of (cx, cy, timestamp) for the initial velocity window.
    _pos_history: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        cx = (self.detection.x1 + self.detection.x2) / 2.0
        cy = (self.detection.y1 + self.detection.y2) / 2.0
        self._last_cx = cx
        self._last_cy = cy
        self._last_update_time = time.time()
        self._pos_history = [(cx, cy, self._last_update_time)]

    # ------------------------------------------------------------------
    def _push_history(self, cx: float, cy: float, now: float) -> None:
        self._pos_history.append((cx, cy, now))
        if len(self._pos_history) > VELOCITY_HISTORY_LEN:
            self._pos_history.pop(0)

    def _recompute_velocity(self, cx: float, cy: float, now: float) -> None:
        """
        Update the EMA velocity estimate given a fresh centre position.
        We use the oldest sample in the history window to get a stable
        direction, then blend it into the running EMA.
        """
        if len(self._pos_history) >= 2:
            oldest_cx, oldest_cy, oldest_t = self._pos_history[0]
            dt = now - oldest_t
            if dt > 1e-4:
                raw_vx = (cx - oldest_cx) / dt
                raw_vy = (cy - oldest_cy) / dt
                self._vx = VELOCITY_ALPHA * raw_vx + (1.0 - VELOCITY_ALPHA) * self._vx
                self._vy = VELOCITY_ALPHA * raw_vy + (1.0 - VELOCITY_ALPHA) * self._vy

    # ------------------------------------------------------------------
    def update(self, new_det: Detection) -> None:
        now = time.time()
        cx  = (new_det.x1 + new_det.x2) / 2.0
        cy  = (new_det.y1 + new_det.y2) / 2.0

        self._push_history(cx, cy, now)
        self._recompute_velocity(cx, cy, now)

        self._last_cx          = cx
        self._last_cy          = cy
        self._last_update_time = now

        self.fused_confidence = 0.7 * new_det.confidence + 0.3 * self.fused_confidence
        self.detection    = new_det
        self.seen_count  += 1
        self.missed_count = 0

    def miss(self) -> None:
        self.missed_count += 1
        self.fused_confidence *= config.PERSISTENCE_DECAY

    # ------------------------------------------------------------------
    def predicted_box(self, now: Optional[float] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Return a predicted bounding box (x1, y1, x2, y2) extrapolated from the
        last known position using the current velocity estimate.

        Returns None if the last real detection is older than MAX_PREDICT_AGE
        (avoids drawing ghost boxes for long-lost objects).
        """
        if now is None:
            now = time.time()

        age = now - self._last_update_time
        if age > MAX_PREDICT_AGE:
            return None

        dt = age  # seconds since last confirmed detection
        det = self.detection
        w   = det.x2 - det.x1
        h   = det.y2 - det.y1

        # Predicted centre
        pred_cx = self._last_cx + self._vx * dt
        pred_cy = self._last_cy + self._vy * dt

        return (
            int(pred_cx - w / 2),
            int(pred_cy - h / 2),
            int(pred_cx + w / 2),
            int(pred_cy + h / 2),
        )

    # ------------------------------------------------------------------
    @property
    def is_confirmed(self) -> bool:
        src      = (self.detection.source or "").lower()
        required = (
            config.PERSISTENCE_REQUIRED_CV_ONLY
            if src.startswith("cv") or "zoomed_cv" in src
            else config.PERSISTENCE_REQUIRED
        )
        return self.seen_count >= required

    @property
    def is_active(self) -> bool:
        return self.missed_count < config.PERSISTENCE_REQUIRED


@dataclass
class _FrameJob:
    frame: np.ndarray
    small: np.ndarray


@dataclass
class _ZoomJob:
    frame: np.ndarray
    box:   Tuple[int, int, int, int]
    source: str


def _scale_det(det: Detection, scale: float) -> Detection:
    """Scale a detection's bounding box coordinates, preserving class metadata."""
    return Detection(
        x1=int(det.x1 * scale), y1=int(det.y1 * scale),
        x2=int(det.x2 * scale), y2=int(det.y2 * scale),
        confidence=det.confidence,
        source=det.source,
        label=det.label,
        class_id=det.class_id,
    )


def _containment(d1: Detection, d2: Detection) -> float:
    """Fraction of the *smaller* box covered by the intersection."""
    ix1 = max(d1.x1, d2.x1); iy1 = max(d1.y1, d2.y1)
    ix2 = min(d1.x2, d2.x2); iy2 = min(d1.y2, d2.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area1 = max(1, (d1.x2 - d1.x1) * (d1.y2 - d1.y1))
    area2 = max(1, (d2.x2 - d2.x1) * (d2.y2 - d2.y1))
    return inter / min(area1, area2)


def _box_area(det: Detection) -> int:
    """Return the pixel area of a detection's bounding box."""
    return max(0, det.x2 - det.x1) * max(0, det.y2 - det.y1)


# =============================================================================
# DETECTION WORKER
# =============================================================================

class DetectionWorker:
    def __init__(self, ai: AIDetector, cv: CVDectector) -> None:
        self.ai = ai
        self.cv = cv
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._zoom_q:  queue.Queue = queue.Queue(maxsize=ZOOM_QUEUE_MAXSIZE)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

        self._ai_dets: List[Detection]      = []
        self._cv_dets: List[Detection]      = []
        self._mask:    Optional[np.ndarray] = None
        self._zoom_results: List[Detection] = []

        # ── Non-blocking result freshness ─────────────────────────────────────
        # Set to True whenever a new batch of detections has been written.
        # FusionEngine reads and clears this flag so it knows when to run the
        # full tracking/fusion pipeline vs. simply re-annotating with predicted
        # positions.
        self._results_fresh: bool = False

        self._thread = threading.Thread(target=self._run, daemon=True, name="detection-worker")
        self._thread.start()

    def push_frame(self, frame: np.ndarray, small: np.ndarray) -> None:
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put_nowait(_FrameJob(frame, small))
        except queue.Full:
            pass

    def push_zoom(self, frame: np.ndarray, box: Tuple[int, int, int, int], source: str) -> None:
        try:
            self._zoom_q.put_nowait(_ZoomJob(frame, box, source))
        except queue.Full:
            pass

    def read_frame(self) -> Tuple[List[Detection], List[Detection], Optional[np.ndarray], bool]:
        """
        Returns (ai_dets, cv_dets, mask, fresh).

        `fresh` is True exactly once per new detection batch — the caller is
        responsible for running the fusion/tracking pipeline only when True.
        """
        with self._lock:
            fresh = self._results_fresh
            self._results_fresh = False
            return self._ai_dets, self._cv_dets, self._mask, fresh

    def read_zoom(self) -> List[Detection]:
        with self._lock:
            out = self._zoom_results
            self._zoom_results = []
            return out

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._process_frame(self._frame_q.get(timeout=0.05))
            except queue.Empty:
                pass
            try:
                self._process_zoom(self._zoom_q.get_nowait())
            except queue.Empty:
                pass

    def _process_frame(self, job: _FrameJob) -> None:
        # Only run AI if the toggle is on
        if is_ai_enabled():
            ai_dets = self.ai.detect(job.small)
        else:
            ai_dets = []

        cv_dets, mask = self.cv.detect(job.small)

        inv     = 1.0 / INFER_SCALE
        ai_dets = [_scale_det(d, inv) for d in ai_dets]
        cv_dets = [_scale_det(d, inv) for d in cv_dets]

        if mask is not None:
            h, w = job.frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        with self._lock:
            self._ai_dets       = ai_dets
            self._cv_dets       = cv_dets
            self._mask          = mask
            self._results_fresh = True   # signal: new batch ready

    def _process_zoom(self, job: _ZoomJob) -> None:
        x1, y1, x2, y2 = job.box
        frame = job.frame
        h, w  = frame.shape[:2]

        pad = max(20, (x2 - x1) // 4)
        rx1 = max(0, x1 - pad); ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad); ry2 = min(h, y2 + pad)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        scale  = config.ZOOM_SCALE_FACTOR
        roi_up = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))

        ai_res = self.ai.detect(roi_up, conf_threshold=config.RECHECK_AI_CONF) if is_ai_enabled() else []
        cv_res, _ = self.cv.detect(roi_up)

        sx = roi.shape[1] / roi_up.shape[1]
        sy = roi.shape[0] / roi_up.shape[0]

        best_det:  Optional[Detection] = None
        best_conf: float = 0.0

        for det in ai_res:
            ox1 = int(rx1 + det.x1 * sx); oy1 = int(ry1 + det.y1 * sy)
            ox2 = int(rx1 + det.x2 * sx); oy2 = int(ry1 + det.y2 * sy)
            scores = self.cv.cv_score_crop(frame, (ox1, oy1, ox2, oy2))
            fused  = 0.6 * det.confidence + 0.4 * scores["total"]
            if fused > best_conf:
                best_conf = fused
                best_det  = Detection(ox1, oy1, ox2, oy2, fused,
                                      f"zoomed_{job.source}",
                                      label=det.label, class_id=det.class_id)

        for det in cv_res:
            ox1 = int(rx1 + det.x1 * sx); oy1 = int(ry1 + det.y1 * sy)
            ox2 = int(rx1 + det.x2 * sx); oy2 = int(ry1 + det.y2 * sy)
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_det  = Detection(ox1, oy1, ox2, oy2, det.confidence,
                                      "zoomed_cv", label="Strawberry", class_id=0)

        if best_det and best_conf >= config.RECHECK_CV_CONF:
            with self._lock:
                self._zoom_results.append(best_det)


# =============================================================================
# FUSION ENGINE
# =============================================================================

class FusionEngine:
    def __init__(self) -> None:
        self._worker = DetectionWorker(AIDetector(), CVDectector())
        self.robot   = RobotController()

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.recheck_counter: Dict[str, int] = defaultdict(int)

        self.frame_count  = 0
        self.detect_count = 0
        self.last_confirmed_hits: List[TrackedObject] = []
        self.last_possible_hits:  List[TrackedObject] = []

        self._last_annotated: Optional[np.ndarray] = None
        self._last_mask:      Optional[np.ndarray] = None
        self._last_debug: Dict = self._make_debug(0, 0, 0, 0, 0, 0)

        # ── State carried between frames for the non-blocking predict path ────
        self._last_ai_dets:    List[Detection]      = []
        self._last_cv_dets:    List[Detection]      = []
        self._last_gripper_xy: Tuple[int, int]      = (0, 0)
        self._last_movement:   str                  = ""
        self._last_target_id:  Optional[int]        = None
        self._last_target_center: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_debug(ai, cv, matches, fused, confirmed, possible) -> Dict:
        return dict(ai_count=ai, cv_count=cv, matches=matches,
                    fused_count=fused, confirmed_count=confirmed, possible_count=possible)

    @staticmethod
    def _is_cv_like(src: str) -> bool:
        return src.startswith("cv") or "zoomed_cv" in src

    @staticmethod
    def _is_ai_like(src: str) -> bool:
        return src.startswith("ai") or "zoomed_ai" in src

    @staticmethod
    def _match_detections(
        ai_dets: List[Detection],
        cv_dets: List[Detection],
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        matches: Dict[int, int] = {}
        used_cv: set            = set()

        for i, ai_det in enumerate(ai_dets):
            if not ai_det.is_targetable:
                continue
            best_score = 0.0
            best_j     = -1
            for j, cv_det in enumerate(cv_dets):
                if j in used_cv or not cv_det.is_targetable:
                    continue
                iou_s  = iou(ai_det, cv_det)
                cont_s = _containment(ai_det, cv_det)
                if iou_s >= config.IOU_MATCH_THRESHOLD or cont_s >= CONTAINMENT_MATCH_THRESHOLD:
                    combined = max(iou_s, cont_s)
                    if combined > best_score:
                        best_score = combined
                        best_j     = j
            if best_j >= 0:
                matches[i] = best_j
                used_cv.add(best_j)

        unmatched_ai = [i for i in range(len(ai_dets)) if i not in matches]
        unmatched_cv = [j for j in range(len(cv_dets)) if j not in used_cv]
        return matches, unmatched_ai, unmatched_cv

    def _request_zoom(self, frame: np.ndarray, box: Tuple[int, int, int, int], source: str) -> None:
        if self.detect_count % RECHECK_EVERY_N_DETECTIONS != 0:
            return
        x1, y1, x2, y2 = box
        key = f"{source}_{(x1+x2)//2 // 50}_{(y1+y2)//2 // 50}"
        if self.recheck_counter[key] >= config.MAX_RECHECKS:
            return
        self.recheck_counter[key] += 1
        self._worker.push_zoom(frame, box, source)

    def _fuse_decision(self, ai_det, cv_det, frame) -> Optional[Detection]:
        if ai_det and cv_det:
            fused = config.YOLO_FUSION_WEIGHT * ai_det.confidence + config.CV_FUSION_WEIGHT * cv_det.confidence
            return Detection(ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2,
                             fused, "fused", label="Strawberry", class_id=0)
        if ai_det:
            if ai_det.confidence > config.HIGH_AI_CONFIDENCE:
                return Detection(ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2,
                                 ai_det.confidence, "ai_high",
                                 label=ai_det.label, class_id=ai_det.class_id)
            self._request_zoom(frame, (ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2), "ai")
            return None
        if cv_det:
            if cv_det.confidence >= config.CV_DIRECT_ACCEPT_THRESHOLD:
                return cv_det
            self._request_zoom(frame, (cv_det.x1, cv_det.y1, cv_det.x2, cv_det.y2), "cv")
            return None
        return None

    def _classify_hits(self, tracked: List[TrackedObject]) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        confirmed: List[TrackedObject] = []
        possible:  List[TrackedObject] = []

        for obj in tracked:
            if obj.is_confirmed:
                confirmed.append(obj)
                continue

            src      = (obj.detection.source or "").lower()
            det_conf = float(obj.detection.confidence or 0.0)

            if self._is_cv_like(src):
                min_seen = config.POSSIBLE_CV_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_CV_ONLY_MIN_CONF
                score    = det_conf
            elif self._is_ai_like(src):
                min_seen = config.POSSIBLE_AI_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_AI_ONLY_MIN_CONF
                score    = det_conf * config.POSSIBLE_AI_CONF_WEIGHT
            else:
                min_seen = config.POSSIBLE_HIT_MIN_SEEN
                min_conf = config.POSSIBLE_HIT_MIN_CONF
                score    = obj.fused_confidence

            if obj.seen_count >= min_seen and score >= min_conf:
                possible.append(obj)

        return confirmed, possible

    def _update_tracking(self, fused_dets: List[Detection]) -> List[TrackedObject]:
        new_tracked: List[TrackedObject] = []
        used: set = set()

        for det in fused_dets:
            if not det.is_targetable:
                continue
            if _box_area(det) < MIN_TARGET_BOX_AREA:
                continue

            best_iou = 0.0
            best_id  = -1
            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used or not obj.is_active:
                    continue
                iou_s  = iou(det, obj.detection)
                cont_s = _containment(det, obj.detection)
                score  = max(iou_s, cont_s)
                if score > best_iou and (
                    iou_s >= config.IOU_MATCH_THRESHOLD
                    or cont_s >= CONTAINMENT_MATCH_THRESHOLD
                ):
                    best_iou = score
                    best_id  = obj_id

            if best_id >= 0:
                current = self.tracked_objects[best_id]
                current.update(det)
                new_tracked.append(current)
                used.add(best_id)
            else:
                obj = TrackedObject(self.next_id, det, fused_confidence=det.confidence)
                self.tracked_objects[self.next_id] = obj
                self.next_id += 1
                new_tracked.append(obj)

        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in used and obj.is_active:
                obj.miss()
                if obj.is_active:
                    new_tracked.append(obj)

        if self.detect_count % CLEANUP_INTERVAL == 0:
            self.tracked_objects = {
                oid: obj for oid, obj in self.tracked_objects.items() if obj.is_active
            }

        return new_tracked

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    @staticmethod
    def draw_annotations(
            frame: np.ndarray,
            ai_dets: List[Detection],
            cv_dets: List[Detection],
            confirmed: List[TrackedObject],
            possible: List[TrackedObject],
            frame_count: int,
            gripper_x: int,
            gripper_y: int,
            target_id: Optional[int],
            target_center: Optional[Tuple[int, int]],
            movement_text: str,
            use_predicted: bool = False,
    ) -> np.ndarray:
        """
        Draw all annotations onto `frame`.

        When `use_predicted` is True (i.e. no fresh detection this frame), the
        confirmed/possible boxes are drawn at their *predicted* positions via
        `TrackedObject.predicted_box()`.  A velocity arrow and fading ghost trail
        show where each strawberry is heading.  Raw AI/CV detector boxes are NOT
        re-drawn on predicted frames — they would just be stale noise.
        """

        out = frame
        now = time.time()

        # ------------------------------------------------------------------
        # Raw detector boxes — only on frames with fresh detections
        # ------------------------------------------------------------------
        if not use_predicted:
            for det in ai_dets:
                color = det.color
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 1)
                cv2.putText(
                    out,
                    f"AI:{det.class_name}:{det.confidence:.2f}",
                    (det.x1, det.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )

            for det in cv_dets:
                color = det.color
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 1)
                cv2.putText(
                    out,
                    f"CV:{det.confidence:.2f}",
                    (det.x1, det.y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )

        # ------------------------------------------------------------------
        # Confirmed targets
        # ------------------------------------------------------------------
        for obj in confirmed:
            det   = obj.detection
            color = (0, 165, 255) if obj.id == target_id else config.COLOR_FUSED
            w     = det.x2 - det.x1
            h     = det.y2 - det.y1

            if use_predicted:
                pbox = obj.predicted_box(now)
                if pbox is None:
                    continue  # too old — skip ghost entirely
                px1, py1, px2, py2 = pbox
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0

                # Ghost trail showing future positions
                _draw_ghost_trail(out, pcx, pcy, obj._vx, obj._vy, w, h, color)

                # Predicted box (thin solid border at current predicted position)
                cv2.rectangle(out, (px1, py1), (px2, py2), color, 1)

                # Velocity arrow from predicted centre
                _draw_velocity_arrow(out, pcx, pcy, obj._vx, obj._vy, color)

                cv2.putText(
                    out,
                    f"#{obj.id} ~{det.confidence:.2f}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
            else:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                # Always draw velocity arrow on confirmed targets even on fresh frames
                # so the user can see motion direction at all times
                cx = (det.x1 + det.x2) / 2.0
                cy = (det.y1 + det.y2) / 2.0
                _draw_velocity_arrow(out, cx, cy, obj._vx, obj._vy, color)
                cv2.putText(
                    out,
                    f"#{obj.id} {det.confidence:.2f}",
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # ------------------------------------------------------------------
        # Possible targets
        # ------------------------------------------------------------------
        for obj in possible:
            det      = obj.detection
            src      = (det.source or "").lower()
            det_conf = float(det.confidence or 0.0)
            w        = det.x2 - det.x1
            h        = det.y2 - det.y1

            if FusionEngine._is_ai_like(src):
                display_score = det_conf * config.POSSIBLE_AI_CONF_WEIGHT
            elif FusionEngine._is_cv_like(src):
                display_score = det_conf
            else:
                display_score = obj.fused_confidence

            if use_predicted:
                pbox = obj.predicted_box(now)
                if pbox is None:
                    continue
                px1, py1, px2, py2 = pbox
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0

                _draw_ghost_trail(out, pcx, pcy, obj._vx, obj._vy, w, h, config.COLOR_POSSIBLE)
                cv2.rectangle(out, (px1, py1), (px2, py2), config.COLOR_POSSIBLE, 1)
                _draw_velocity_arrow(out, pcx, pcy, obj._vx, obj._vy, config.COLOR_POSSIBLE)
                cv2.putText(
                    out,
                    f"P#{obj.id} ~{display_score:.2f}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    config.COLOR_POSSIBLE,
                    1,
                )
            else:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), config.COLOR_POSSIBLE, 1)
                cx = (det.x1 + det.x2) / 2.0
                cy = (det.y1 + det.y2) / 2.0
                _draw_velocity_arrow(out, cx, cy, obj._vx, obj._vy, config.COLOR_POSSIBLE)
                cv2.putText(
                    out,
                    f"P#{obj.id} {display_score:.2f}",
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    config.COLOR_POSSIBLE,
                    1,
                )

        # ------------------------------------------------------------------
        # Gripper bounding box
        # ------------------------------------------------------------------
        gripping = False
        try:
            import gripper as _gripper
            state    = _gripper.get_state().lower()
            gripping = "grip" in state
        except Exception:
            pass

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
            RobotController.get_gripper_bbox(gripper_x, gripper_y)
        )

        contained = False
        if target_center is not None:
            for obj in confirmed + possible:
                if obj.id == target_id:
                    det_for_check = obj.detection
                    if use_predicted:
                        pbox = obj.predicted_box(now)
                        if pbox:
                            # Build a temporary Detection-like object from predicted box
                            det_for_check = Detection(
                                *pbox,
                                confidence=obj.detection.confidence,
                                source=obj.detection.source,
                                label=obj.detection.label,
                                class_id=obj.detection.class_id,
                            )
                    contained = RobotController.is_detection_fully_contained(
                        det_for_check,
                        (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                    )
                    break

        if gripping:
            gripper_color = (0, 0, 255)
        elif contained:
            gripper_color = (0, 255, 0)
        else:
            gripper_color = (255, 0, 255)

        cv2.rectangle(out, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), gripper_color, 2)
        cv2.circle(out, (gripper_x, gripper_y), 8, gripper_color, -1)
        cv2.putText(
            out,
            "GRIPPER",
            (gripper_x + 10, gripper_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            gripper_color,
            2,
        )

        if target_center is not None:
            cv2.line(out, (gripper_x, gripper_y), target_center, (0, 165, 255), 2)

        # ------------------------------------------------------------------
        # Legend
        # ------------------------------------------------------------------
        legend_items = [
            (CLASS_COLORS[0], "Strawberry (targetable)"),
            (CLASS_COLORS[1], "Rotten (display only)"),
            (CLASS_COLORS[2], "Leaf (display only)"),
        ]
        lx, ly = 10, frame.shape[0] - 10 - len(legend_items) * 20
        for color, label in legend_items:
            cv2.circle(out, (lx + 6, ly + 6), 5, color, -1)
            cv2.putText(out, label, (lx + 16, ly + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            ly += 20

        # ------------------------------------------------------------------
        # HUD
        # ------------------------------------------------------------------
        ai_badge   = "AI:ON" if is_ai_enabled() else "AI:OFF"
        pred_badge = " [PRED]" if use_predicted else ""
        cv2.putText(
            out,
            f"Frame {frame_count} | Hits:{len(confirmed)} | Possible:{len(possible)} | {ai_badge}{pred_badge}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Robot: {movement_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[TrackedObject], Dict, Optional[np.ndarray]]:
        self.frame_count += 1

        small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE,
                           interpolation=cv2.INTER_LINEAR)
        self._worker.push_frame(frame, small)

        gripper_x = frame.shape[1] // 2
        gripper_y = frame.shape[0] // 2

        # ── Check if fresh detections are available (non-blocking) ───────────
        ai_dets, cv_dets, mask, fresh = self._worker.read_frame()

        if not fresh:
            # ── PREDICT PATH — no new detections yet; return immediately ─────
            # Re-draw the current frame with predicted bounding boxes so the
            # display never freezes waiting for the detection thread.
            annotated = self.draw_annotations(
                frame.copy(),
                self._last_ai_dets,
                self._last_cv_dets,
                self.last_confirmed_hits,
                self.last_possible_hits,
                self.frame_count,
                gripper_x, gripper_y,
                self._last_target_id,
                self._last_target_center,
                self._last_movement,
                use_predicted=True,
            )
            return annotated, self.last_confirmed_hits, self._last_debug, self._last_mask

        # ── DETECTION PATH — fresh batch from the worker ─────────────────────
        self.detect_count += 1
        zoom_dets = self._worker.read_zoom()

        self._last_ai_dets = ai_dets
        self._last_cv_dets = cv_dets

        # Split AI detections: targetable vs display-only
        ai_targetable   = [d for d in ai_dets if d.is_targetable]
        ai_display_only = [d for d in ai_dets if not d.is_targetable]  # noqa: F841

        if is_ai_enabled():
            matches, unmatched_ai, unmatched_cv = self._match_detections(ai_targetable, cv_dets)
        else:
            matches      = {}
            unmatched_ai = []
            unmatched_cv = list(range(len(cv_dets)))

        fused: List[Detection] = [d for d in zoom_dets if d.is_targetable]

        for ai_idx, cv_idx in matches.items():
            maybe = self._fuse_decision(ai_targetable[ai_idx], cv_dets[cv_idx], frame)
            if maybe:
                fused.append(maybe)

        for ai_idx in unmatched_ai:
            det = ai_targetable[ai_idx]
            if det.confidence < config.LOW_AI_CONFIDENCE:
                self._request_zoom(frame, (det.x1, det.y1, det.x2, det.y2), "ai_low")
            else:
                maybe = self._fuse_decision(det, None, frame)
                if maybe:
                    fused.append(maybe)

        for cv_idx in unmatched_cv:
            maybe = self._fuse_decision(None, cv_dets[cv_idx], frame)
            if maybe:
                fused.append(maybe)

        tracked  = self._update_tracking(fused)
        confirmed, possible = self._classify_hits(tracked)
        self.last_confirmed_hits = confirmed
        self.last_possible_hits  = possible

        # ── Target selection ──────────────────────────────────────────────────
        target_pool = [obj.detection for obj in confirmed]
        using_possible_fallback = False

        if not target_pool and config.POSSIBLE_TARGET_FALLBACK_ENABLED:
            def possible_score(p: TrackedObject) -> float:
                src      = (p.detection.source or "").lower()
                det_conf = float(p.detection.confidence or 0.0)
                if self._is_cv_like(src):   return det_conf
                if self._is_ai_like(src):   return det_conf * config.POSSIBLE_AI_CONF_WEIGHT
                return p.fused_confidence

            target_pool = [
                obj.detection for obj in possible
                if possible_score(obj) >= config.POSSIBLE_TARGET_MIN_CONF
            ]
            using_possible_fallback = bool(target_pool)

        target   = self.robot.choose_target(target_pool, gripper_x, gripper_y)
        movement = self.robot.generate_movementstring(gripper_x, gripper_y)

        dx = self.robot.generate_dx(gripper_x)
        dy = self.robot.generate_dy(gripper_y)

        mode = "possible" if using_possible_fallback else "confirmed"
        print(f"[ROBOT][{mode}] {movement}: X{dx}, Y{dy}, {self.robot.generate_depthstring()}")

        self.robot.drive_hardware(gripper_x, gripper_y)

        # ── Resolve target id / centre ────────────────────────────────────────
        target_id     = None
        target_center = None
        if target is not None:
            target_center = (target.center_x, target.center_y)
            search_list   = confirmed if not using_possible_fallback else confirmed + possible
            for obj in search_list:
                if obj.detection == target.detection:
                    target_id = obj.id
                    break

        # Cache for predict path
        self._last_movement      = movement
        self._last_target_id     = target_id
        self._last_target_center = target_center

        self._last_debug = self._make_debug(
            len(ai_dets), len(cv_dets), len(matches),
            len(fused), len(confirmed), len(possible),
        )
        self._last_mask = mask

        annotated = self.draw_annotations(
            frame,
            ai_dets,
            cv_dets,
            confirmed,
            possible,
            self.frame_count,
            gripper_x, gripper_y,
            target_id, target_center,
            movement,
            use_predicted=False,
        )

        return annotated, confirmed, self._last_debug, mask

    def shutdown(self) -> None:
        self._worker.stop()


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

# How far ahead (seconds) the velocity arrow tip represents.
# E.g. 0.5 means "where the strawberry will be in 500 ms at current speed".
ARROW_LOOKAHEAD_SEC: float = 0.5

# Minimum pixel length for the arrow to be drawn (avoids drawing tiny arrows
# for nearly-stationary strawberries).
ARROW_MIN_PX: int = 8

# Ghost trail: how many evenly-spaced future positions to show as fading dots.
TRAIL_STEPS: int  = 4
TRAIL_MAX_SEC: float = 0.5   # time span covered by the trail dots


def _draw_velocity_arrow(
    img: np.ndarray,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    color: Tuple[int, int, int],
    lookahead: float = ARROW_LOOKAHEAD_SEC,
    min_px: int      = ARROW_MIN_PX,
) -> None:
    """
    Draw an arrow from the current centre (cx, cy) pointing in the direction
    of motion.  Arrow length is proportional to speed × lookahead seconds.
    """
    tip_x = cx + vx * lookahead
    tip_y = cy + vy * lookahead
    length = np.hypot(tip_x - cx, tip_y - cy)
    if length < min_px:
        return

    src = (int(cx), int(cy))
    tip = (int(tip_x), int(tip_y))

    # Thick white outline first so the arrow is readable on any background
    cv2.arrowedLine(img, src, tip, (255, 255, 255), thickness=4, tipLength=0.25)
    cv2.arrowedLine(img, src, tip, color,           thickness=2, tipLength=0.25)

    # Speed label next to the tip (pixels per second)
    speed_px_s = np.hypot(vx, vy)
    cv2.putText(
        img,
        f"{speed_px_s:.0f}px/s",
        (tip[0] + 4, tip[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
    )


def _draw_ghost_trail(
    img: np.ndarray,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    steps: int      = TRAIL_STEPS,
    max_sec: float  = TRAIL_MAX_SEC,
) -> None:
    """
    Draw a series of fading semi-transparent ghost rectangles ahead of the
    strawberry to visualise the predicted trajectory.

    Each successive ghost is drawn more transparently and slightly smaller
    so the visual weight tapers off into the future.
    """
    overlay = img.copy()
    for i in range(1, steps + 1):
        t      = max_sec * i / steps          # time offset for this ghost
        alpha  = 0.35 * (1.0 - i / (steps + 1))  # fade with distance
        scale  = 1.0 - 0.06 * i              # shrink slightly
        gcx    = cx + vx * t
        gcy    = cy + vy * t
        gw     = int(w * scale)
        gh     = int(h * scale)
        gx1    = int(gcx - gw / 2)
        gy1    = int(gcy - gh / 2)
        gx2    = int(gcx + gw / 2)
        gy2    = int(gcy + gh / 2)

        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, 1)
        # Blend just the affected pixels back onto img
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        overlay = img.copy()   # refresh overlay base for next iteration

        # Small dot at predicted centre
        cv2.circle(img, (int(gcx), int(gcy)), 3, color, -1)
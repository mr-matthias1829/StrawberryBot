"""
fusion_engine.py  — reworked
==============================
Major changes vs. the old version:

1. TRACKING  — real per-track Kalman filter (constant-velocity, 8-state)
   replaces the old EMA + IoU-only matching.  Tracks coast through missed
   frames instead of flickering in and out.  Matching uses both IoU AND
   normalised centre-distance so fast-moving boxes keep their lock.

2. ZOOM RECHECK  — keyed per *track id* with a time-based cooldown.
   Results are merged back into the track that requested them (matched by
   IoU against the predicted box), not dropped as disconnected one-offs.

3. AI PRIMACY  — three-tier acceptance:
     HIGH  → trusted outright (CV is bonus)
     MED   → trusted unless CV actively disagrees
     LOW   → needs CV corroboration or goes to recheck
   CV-only still works as a fallback but is held to a stricter direct-accept bar.

4. FRAME-AGE DEBOUNCE  — every detection carries the monotonic timestamp of
   the frame it came from.  Detections older than MAX_ACCEPTABLE_FRAME_AGE_S
   are excluded from driving the robot, addressing camera latency directly.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
import servo_status
from detection import AIDetector, CVDectector, Detection, CLASS_COLORS, CLASS_NAMES, iou
from robot_controller import RobotController


INFER_SCALE      = 1
DETECT_EVERY     = 1
CLEANUP_INTERVAL = 30
MIN_TARGET_BOX_AREA = 900   # px²

ARROW_LOOKAHEAD_SEC = 0.5
ARROW_MIN_PX        = 8
TRAIL_STEPS         = 4
TRAIL_MAX_SEC       = 0.5


# ---------------------------------------------------------------------------
# Detector toggles
# ---------------------------------------------------------------------------

_ai_enabled      = True
_ai_enabled_lock = threading.Lock()

def set_ai_enabled(enabled: bool) -> None:
    global _ai_enabled
    with _ai_enabled_lock:
        _ai_enabled = bool(enabled)
    print(f"[FusionEngine] AI {'ENABLED' if enabled else 'DISABLED'}")

def is_ai_enabled() -> bool:
    with _ai_enabled_lock:
        return _ai_enabled


_cv_enabled      = True
_cv_enabled_lock = threading.Lock()

def set_cv_enabled(enabled: bool) -> None:
    global _cv_enabled
    with _cv_enabled_lock:
        _cv_enabled = bool(enabled)
    print(f"[FusionEngine] CV {'ENABLED' if enabled else 'DISABLED'}")

def is_cv_enabled() -> bool:
    with _cv_enabled_lock:
        return _cv_enabled


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _scale_det(det: Detection, scale: float) -> Detection:
    return Detection(
        x1=int(det.x1 * scale), y1=int(det.y1 * scale),
        x2=int(det.x2 * scale), y2=int(det.y2 * scale),
        confidence=det.confidence, source=det.source,
        label=det.label, class_id=det.class_id, timestamp=det.timestamp,
    )

def _containment(d1: Detection, d2: Detection) -> float:
    ix1, iy1 = max(d1.x1, d2.x1), max(d1.y1, d2.y1)
    ix2, iy2 = min(d1.x2, d2.x2), min(d1.y2, d2.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    return inter / min(
        max(1, (d1.x2 - d1.x1) * (d1.y2 - d1.y1)),
        max(1, (d2.x2 - d2.x1) * (d2.y2 - d2.y1)),
    )

def _box_area(det: Detection) -> int:
    return max(0, det.x2 - det.x1) * max(0, det.y2 - det.y1)

def _box_diag(x1, y1, x2, y2) -> float:
    return max(1.0, np.hypot(x2 - x1, y2 - y1))

def _center_dist_norm(a: Detection, b: Detection) -> float:
    """Centre distance normalised by average box diagonal — scale-invariant."""
    acx, acy = a.center
    bcx, bcy = b.center
    dist = np.hypot(acx - bcx, acy - bcy)
    diag = (_box_diag(a.x1, a.y1, a.x2, a.y2) + _box_diag(b.x1, b.y1, b.x2, b.y2)) / 2.0
    return dist / diag


# =============================================================================
# Kalman Box Track
# =============================================================================

class KalmanBoxTrack:
    """
    Single-object track — constant-velocity Kalman filter over
    state [cx, cy, w, h, vx, vy, vw, vh].

    Using a proper predict/update cycle lets the track coast convincingly
    through missed frames (the main flicker fix).
    """

    _next_id = 1

    def __init__(self, det: Detection) -> None:
        self.id = KalmanBoxTrack._next_id
        KalmanBoxTrack._next_id += 1

        self.kf = cv2.KalmanFilter(8, 4)

        # Measurement matrix: observe [cx, cy, w, h] from state
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        q_pos = config.KALMAN_PROCESS_NOISE_POS
        q_vel = config.KALMAN_PROCESS_NOISE_VEL
        self.kf.processNoiseCov = np.diag(
            [q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel]
        ).astype(np.float32)

        self._set_measurement_noise(det.source)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

        cx, cy = det.center
        w = max(1, det.x2 - det.x1)
        h = max(1, det.y2 - det.y1)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(8, 1)

        self.detection        = det
        self.last_update_time = det.timestamp
        self.last_predict_time= det.timestamp
        self.created_at       = det.timestamp

        self.update_count     = 1
        self.coast_seconds    = 0.0

        self.last_zoom_request_time = 0.0
        self.zoom_request_count     = 0

        self.fused_confidence = det.confidence
        self.source_history: List[str] = [det.source]

    # ------------------------------------------------------------------

    def _set_measurement_noise(self, source: str) -> None:
        noise = (config.KALMAN_MEASUREMENT_NOISE_AI
                 if source.startswith("ai") else config.KALMAN_MEASUREMENT_NOISE_CV)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * noise

    def predict(self, now: float) -> Tuple[int, int, int, int]:
        dt = max(0.0, now - self.last_predict_time)
        self.last_predict_time = now

        F = np.eye(8, dtype=np.float32)
        F[0, 4] = dt; F[1, 5] = dt; F[2, 6] = dt; F[3, 7] = dt
        self.kf.transitionMatrix = F

        state = self.kf.predict()
        cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        w = max(2.0, w); h = max(2.0, h)
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    def update(self, det: Detection) -> None:
        cx, cy = det.center
        w = max(1, det.x2 - det.x1)
        h = max(1, det.y2 - det.y1)
        meas = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)

        self._set_measurement_noise(det.source)
        self.kf.correct(meas)

        self.detection     = det
        self.last_update_time = det.timestamp
        self.coast_seconds = 0.0
        self.update_count += 1
        self.fused_confidence = 0.6 * det.confidence + 0.4 * self.fused_confidence

        self.source_history.append(det.source)
        if len(self.source_history) > 8:
            self.source_history.pop(0)

    def mark_missed(self, now: float) -> None:
        self.coast_seconds = now - self.last_update_time

    # ------------------------------------------------------------------
    # Properties

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[4, 0]), float(s[5, 0])

    @property
    def predicted_box_now(self) -> Tuple[int, int, int, int]:
        s = self.kf.statePost
        cx, cy, w, h = s[0, 0], s[1, 0], s[2, 0], s[3, 0]
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    @property
    def is_lost(self) -> bool:
        return self.coast_seconds > config.TRACK_COAST_MAX_S

    @property
    def is_confirmed(self) -> bool:
        tracked_for = self.last_update_time - self.created_at
        return (self.update_count >= config.TRACK_CONFIRM_MIN_UPDATES and
                tracked_for     >= config.TRACK_CONFIRM_TIME_S)

    @property
    def frame_age(self) -> float:
        return self.detection.age()

    def can_request_zoom(self, now: float) -> bool:
        if self.zoom_request_count >= config.ZOOM_RECHECK_MAX_PER_TRACK:
            if now - self.last_zoom_request_time > config.ZOOM_RECHECK_COOLDOWN_S * 4:
                self.zoom_request_count = max(0, self.zoom_request_count - 2)
            else:
                return False
        return (now - self.last_zoom_request_time) >= config.ZOOM_RECHECK_COOLDOWN_S

    def mark_zoom_requested(self, now: float) -> None:
        self.last_zoom_request_time = now
        self.zoom_request_count    += 1


# =============================================================================
# Worker: detection off the main thread + zoom rechecks
# =============================================================================

@dataclass
class _FrameJob:
    frame:     np.ndarray
    small:     np.ndarray
    timestamp: float

@dataclass
class _ZoomJob:
    frame:    np.ndarray
    box:      Tuple[int, int, int, int]
    source:   str
    track_id: int
    timestamp:float

@dataclass
class _ZoomResult:
    detection: Detection
    track_id:  int


class DetectionWorker:
    def __init__(self, ai: AIDetector, cv: CVDectector) -> None:
        self.ai = ai
        self.cv = cv
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._zoom_q:  queue.Queue = queue.Queue(maxsize=8)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

        self._ai_dets:      List[Detection]   = []
        self._cv_dets:      List[Detection]   = []
        self._mask:         Optional[np.ndarray] = None
        self._zoom_results: List[_ZoomResult] = []
        self._results_fresh = False

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="detection-worker"
        )
        self._thread.start()

    def push_frame(self, frame: np.ndarray, small: np.ndarray, timestamp: float) -> None:
        try:   self._frame_q.get_nowait()
        except queue.Empty: pass
        try:   self._frame_q.put_nowait(_FrameJob(frame, small, timestamp))
        except queue.Full:  pass

    def push_zoom(self, frame: np.ndarray, box, source: str,
                  track_id: int, timestamp: float) -> None:
        try:
            self._zoom_q.put_nowait(_ZoomJob(frame, box, source, track_id, timestamp))
        except queue.Full:
            pass

    def read_frame(self):
        with self._lock:
            fresh = self._results_fresh
            self._results_fresh = False
            return self._ai_dets, self._cv_dets, self._mask, fresh

    def read_zoom(self) -> List[_ZoomResult]:
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
            for _ in range(2):
                try:
                    self._process_zoom(self._zoom_q.get_nowait())
                except queue.Empty:
                    break

    def _process_frame(self, job: _FrameJob) -> None:
        ai_dets = (self.ai.detect(job.small, timestamp=job.timestamp)
                   if is_ai_enabled() else [])
        cv_dets, mask = (self.cv.detect(job.small, timestamp=job.timestamp)
                         if is_cv_enabled() else ([], None))

        inv = 1.0 / INFER_SCALE
        ai_dets = [_scale_det(d, inv) for d in ai_dets]
        cv_dets = [_scale_det(d, inv) for d in cv_dets]

        if mask is not None:
            h, w = job.frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        with self._lock:
            self._ai_dets       = ai_dets
            self._cv_dets       = cv_dets
            self._mask          = mask
            self._results_fresh = True

    def _process_zoom(self, job: _ZoomJob) -> None:
        x1, y1, x2, y2 = job.box
        h, w = job.frame.shape[:2]
        pad  = max(20, (x2 - x1) // 4)
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
        roi = job.frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        scale  = config.ZOOM_SCALE_FACTOR
        roi_up = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))

        ai_res = (self.ai.detect(roi_up, conf_threshold=config.RECHECK_AI_CONF,
                                 timestamp=job.timestamp)
                  if is_ai_enabled() else [])
        cv_res, _ = (self.cv.detect(roi_up, timestamp=job.timestamp)
                     if is_cv_enabled() else ([], None))

        sx = roi.shape[1] / roi_up.shape[1]
        sy = roi.shape[0] / roi_up.shape[0]

        best_det, best_conf = None, 0.0

        for det in ai_res:
            ox1 = int(rx1 + det.x1 * sx); oy1 = int(ry1 + det.y1 * sy)
            ox2 = int(rx1 + det.x2 * sx); oy2 = int(ry1 + det.y2 * sy)
            cv_score = self.cv.cv_score_crop(job.frame, (ox1, oy1, ox2, oy2))["total"]
            fused = (config.AI_DOMINANT_FUSION_WEIGHT * det.confidence
                     + config.CV_MINORITY_FUSION_WEIGHT * cv_score)
            if fused > best_conf:
                best_conf = fused
                best_det  = Detection(ox1, oy1, ox2, oy2, fused,
                                      f"zoomed_{job.source}",
                                      label=det.label, class_id=det.class_id,
                                      timestamp=job.timestamp)

        if cv_res:
            for det in cv_res:
                ox1 = int(rx1 + det.x1 * sx); oy1 = int(ry1 + det.y1 * sy)
                ox2 = int(rx1 + det.x2 * sx); oy2 = int(ry1 + det.y2 * sy)
                if det.confidence > best_conf:
                    best_conf = det.confidence
                    best_det  = Detection(ox1, oy1, ox2, oy2, det.confidence, "zoomed_cv",
                                          label="Strawberry", class_id=0,
                                          timestamp=job.timestamp)

        if best_det and best_conf >= config.RECHECK_CV_CONF:
            with self._lock:
                self._zoom_results.append(_ZoomResult(best_det, job.track_id))


# =============================================================================
# Drawing utilities
# =============================================================================

def _draw_velocity_arrow(img, cx, cy, vx, vy, color,
                         lookahead=ARROW_LOOKAHEAD_SEC, min_px=ARROW_MIN_PX):
    tip_x, tip_y = cx + vx * lookahead, cy + vy * lookahead
    if np.hypot(tip_x - cx, tip_y - cy) < min_px:
        return
    src, tip = (int(cx), int(cy)), (int(tip_x), int(tip_y))
    cv2.arrowedLine(img, src, tip, (255, 255, 255), thickness=4, tipLength=0.25)
    cv2.arrowedLine(img, src, tip, color,           thickness=2, tipLength=0.25)
    cv2.putText(img, f"{np.hypot(vx, vy):.0f}px/s", (tip[0] + 4, tip[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

def _draw_ghost_trail(img, cx, cy, vx, vy, w, h, color,
                      steps=TRAIL_STEPS, max_sec=TRAIL_MAX_SEC):
    overlay = img.copy()
    for i in range(1, steps + 1):
        t     = max_sec * i / steps
        alpha = 0.35 * (1.0 - i / (steps + 1))
        scale = 1.0 - 0.06 * i
        gcx, gcy = cx + vx * t, cy + vy * t
        gw, gh   = int(w * scale), int(h * scale)
        cv2.rectangle(overlay,
                      (int(gcx - gw / 2), int(gcy - gh / 2)),
                      (int(gcx + gw / 2), int(gcy + gh / 2)), color, 1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        overlay = img.copy()
        cv2.circle(img, (int(gcx), int(gcy)), 3, color, -1)


# =============================================================================
# Fusion Engine
# =============================================================================

class FusionEngine:
    def __init__(self) -> None:
        self._worker = DetectionWorker(AIDetector(), CVDectector())
        self.robot   = RobotController()
        self.tracks: Dict[int, KalmanBoxTrack] = {}
        self.frame_count  = 0
        self.detect_count = 0

        self.last_confirmed_hits: List[KalmanBoxTrack] = []
        self.last_possible_hits:  List[KalmanBoxTrack] = []
        self._last_ai_dets:  List[Detection]      = []
        self._last_cv_dets:  List[Detection]      = []
        self._last_mask:     Optional[np.ndarray] = None
        self._last_movement  = ""
        self._last_target_id: Optional[int]       = None
        self._last_target_center: Optional[Tuple[int, int]] = None
        self._last_debug = self._make_debug(0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def _make_debug(ai, cv, matches, fused, confirmed, possible, stale) -> Dict:
        return dict(ai_count=ai, cv_count=cv, matches=matches, fused_count=fused,
                    confirmed_count=confirmed, possible_count=possible, stale_count=stale)

    # ------------------------------------------------------------------
    # Per-frame AI/CV matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_ai_cv(ai_dets: List[Detection], cv_dets: List[Detection]):
        matches: Dict[int, int] = {}
        used_cv: set = set()
        for i, ai_det in enumerate(ai_dets):
            if not ai_det.is_targetable:
                continue
            best_score, best_j = 0.0, -1
            for j, cv_det in enumerate(cv_dets):
                if j in used_cv or not cv_det.is_targetable:
                    continue
                iou_s  = iou(ai_det, cv_det)
                cont_s = _containment(ai_det, cv_det)
                if iou_s >= config.TRACK_MATCH_IOU_MIN or cont_s >= 0.35:
                    combined = max(iou_s, cont_s)
                    if combined > best_score:
                        best_score, best_j = combined, j
            if best_j >= 0:
                matches[i] = best_j
                used_cv.add(best_j)
        return (matches,
                [i for i in range(len(ai_dets)) if i not in matches],
                [j for j in range(len(cv_dets)) if j not in used_cv])

    # ------------------------------------------------------------------
    # Acceptance logic — AI primacy lives here
    # ------------------------------------------------------------------

    def _accept_ai_cv_pair(self, ai_det: Detection,
                           cv_det: Optional[Detection]) -> Optional[Detection]:
        conf     = ai_det.confidence
        cv_score = cv_det.confidence if cv_det is not None else 0.0

        if conf >= config.AI_TRUST_HIGH:
            fused_conf = conf if cv_det is None else (
                config.AI_DOMINANT_FUSION_WEIGHT * conf
                + config.CV_MINORITY_FUSION_WEIGHT * cv_score)
            return Detection(ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2,
                             fused_conf,
                             "ai_high" if cv_det is None else "fused",
                             label=ai_det.label, class_id=ai_det.class_id,
                             timestamp=ai_det.timestamp)

        if conf >= config.AI_TRUST_MED:
            if cv_det is not None and cv_score < config.AI_TRUST_MED_CV_VETO:
                return None   # CV disagrees → send to recheck
            fused_conf = conf if cv_det is None else (
                config.AI_DOMINANT_FUSION_WEIGHT * conf
                + config.CV_MINORITY_FUSION_WEIGHT * cv_score)
            return Detection(ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2,
                             fused_conf, "fused",
                             label=ai_det.label, class_id=ai_det.class_id,
                             timestamp=ai_det.timestamp)

        if conf >= config.AI_TRUST_LOW:
            if cv_det is not None and cv_score >= config.AI_TRUST_LOW_CV_CONFIRM:
                fused_conf = (config.AI_DOMINANT_FUSION_WEIGHT * conf
                              + config.CV_MINORITY_FUSION_WEIGHT * cv_score)
                return Detection(ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2,
                                 fused_conf, "fused",
                                 label=ai_det.label, class_id=ai_det.class_id,
                                 timestamp=ai_det.timestamp)
            return None   # needs zoom recheck

        return None   # below LOW → recheck

    def _accept_cv_only(self, cv_det: Detection) -> Optional[Detection]:
        if cv_det.confidence >= config.CV_DIRECT_ACCEPT_THRESHOLD:
            return cv_det
        return None

    # ------------------------------------------------------------------
    # Zoom request helper — keyed by track, with cooldown decay
    # ------------------------------------------------------------------

    def _request_zoom_for_track(self, track: Optional[KalmanBoxTrack],
                                 frame: np.ndarray,
                                 box: Tuple[int, int, int, int],
                                 source: str, now: float) -> None:
        if track is not None:
            if not track.can_request_zoom(now):
                return
            track.mark_zoom_requested(now)
            self._worker.push_zoom(frame, box, source, track.id, now)
        else:
            self._worker.push_zoom(frame, box, source, -1, now)

    # ------------------------------------------------------------------
    # Tracking: associate detections → Kalman tracks
    # ------------------------------------------------------------------

    def _associate(self, fused_dets: List[Detection],
                   now: float) -> Dict[int, int]:
        """Return {detection_index: track_id} for matched pairs."""
        predicted: Dict[int, Tuple[int, int, int, int]] = {
            tid: t.predict(now) for tid, t in self.tracks.items()
        }

        candidates = []
        for di, det in enumerate(fused_dets):
            for tid, pbox in predicted.items():
                track = self.tracks[tid]
                if track.detection.class_id != det.class_id:
                    continue
                px1, py1, px2, py2 = pbox
                pseudo  = Detection(px1, py1, px2, py2, 1.0, "pred",
                                    class_id=det.class_id)
                iou_s   = iou(det, pseudo)
                dist_n  = _center_dist_norm(det, pseudo)
                if (iou_s  >= config.TRACK_MATCH_IOU_MIN or
                        dist_n <= config.TRACK_MATCH_CENTER_DIST_MAX):
                    score = iou_s + max(0.0, 1.0 - dist_n) * 0.5
                    candidates.append((score, di, tid))

        candidates.sort(key=lambda c: c[0], reverse=True)
        matches: Dict[int, int] = {}
        used_dets: set = set()
        used_tracks: set = set()
        for score, di, tid in candidates:
            if di in used_dets or tid in used_tracks:
                continue
            matches[di] = tid
            used_dets.add(di)
            used_tracks.add(tid)
        return matches

    def _update_tracking(self, fused_dets: List[Detection],
                         now: float) -> List[KalmanBoxTrack]:
        valid = [d for d in fused_dets
                 if d.is_targetable and _box_area(d) >= MIN_TARGET_BOX_AREA]

        matches = self._associate(valid, now)

        for di, det in enumerate(valid):
            tid = matches.get(di)
            if tid is not None:
                self.tracks[tid].update(det)
            else:
                t = KalmanBoxTrack(det)
                self.tracks[t.id] = t

        matched_tids = set(matches.values())
        for tid, track in self.tracks.items():
            if tid not in matched_tids:
                track.mark_missed(now)

        # Periodic + emergency cleanup
        if self.detect_count % CLEANUP_INTERVAL == 0 or len(self.tracks) > 40:
            self.tracks = {tid: t for tid, t in self.tracks.items() if not t.is_lost}
        else:
            dead = [tid for tid, t in self.tracks.items()
                    if t.coast_seconds > config.TRACK_COAST_MAX_S * 3]
            for tid in dead:
                del self.tracks[tid]

        return [t for t in self.tracks.values() if not t.is_lost]

    def _classify_hits(self, tracks: List[KalmanBoxTrack], now: float):
        confirmed, possible = [], []
        for t in tracks:
            if t.is_confirmed:
                confirmed.append(t)
            elif t.update_count >= 1:
                possible.append(t)
        return confirmed, possible

    # ------------------------------------------------------------------
    # Main per-frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        try:
            from web_server import is_killed
            if is_killed():
                return frame.copy(), [], self._last_debug, self._last_mask
        except ImportError:
            pass

        now = time.monotonic()
        self.frame_count += 1
        small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE,
                           interpolation=cv2.INTER_LINEAR)
        self._worker.push_frame(frame, small, now)
        gripper_x = frame.shape[1] // 2
        gripper_y = frame.shape[0] // 2

        ai_dets, cv_dets, mask, fresh = self._worker.read_frame()

        if not fresh:
            # No new detector output this tick — predict tracks forward
            for t in self.tracks.values():
                t.predict(now)
            confirmed, possible = self._classify_hits(
                [t for t in self.tracks.values() if not t.is_lost], now)
            annotated = self.draw_annotations(
                frame.copy(), self._last_ai_dets, self._last_cv_dets,
                confirmed, possible, self.frame_count, gripper_x, gripper_y,
                self._last_target_id, self._last_target_center,
                self._last_movement, use_predicted=True,
            )
            return annotated, confirmed, self._last_debug, self._last_mask

        # ── We have fresh detections ──────────────────────────────────

        self.detect_count += 1
        zoom_results = self._worker.read_zoom()
        self._last_ai_dets = ai_dets
        self._last_cv_dets = cv_dets

        ai_targetable = [d for d in ai_dets if d.is_targetable]
        if is_ai_enabled():
            matches, unmatched_ai, unmatched_cv = self._match_ai_cv(ai_targetable, cv_dets)
        else:
            matches, unmatched_ai, unmatched_cv = {}, [], list(range(len(cv_dets)))

        fused: List[Detection] = []
        recheck_requests: List[Tuple[Tuple[int, int, int, int], str]] = []

        for ai_idx, cv_idx in matches.items():
            accepted = self._accept_ai_cv_pair(ai_targetable[ai_idx], cv_dets[cv_idx])
            if accepted:
                fused.append(accepted)
            else:
                det = ai_targetable[ai_idx]
                recheck_requests.append(((det.x1, det.y1, det.x2, det.y2), "ai"))

        for ai_idx in unmatched_ai:
            det      = ai_targetable[ai_idx]
            accepted = self._accept_ai_cv_pair(det, None)
            if accepted:
                fused.append(accepted)
            else:
                recheck_requests.append(((det.x1, det.y1, det.x2, det.y2), "ai"))

        for cv_idx in unmatched_cv:
            det      = cv_dets[cv_idx]
            accepted = self._accept_cv_only(det)
            if accepted:
                fused.append(accepted)
            else:
                recheck_requests.append(((det.x1, det.y1, det.x2, det.y2), "cv"))

        # Merge zoom results back into tracks
        for zr in zoom_results:
            if zr.detection.is_targetable:
                fused.append(zr.detection)

        tracks    = self._update_tracking(fused, now)
        confirmed, possible = self._classify_hits(tracks, now)
        self.last_confirmed_hits = confirmed
        self.last_possible_hits  = possible

        # Fire per-track zoom rechecks for anything that didn't pass acceptance
        for box, source in recheck_requests:
            bx1, by1, bx2, by2 = box
            best_track, best_iou_s = None, 0.0
            probe = Detection(bx1, by1, bx2, by2, 1.0, source, class_id=0)
            for t in tracks:
                s = iou(probe, t.detection)
                if s > best_iou_s:
                    best_iou_s, best_track = s, t
            self._request_zoom_for_track(
                best_track if best_iou_s >= 0.2 else None,
                frame, box, source, now)

        # ── Frame-age filter: only fresh detections drive the robot ──
        fresh_confirmed = [t for t in confirmed
                           if t.frame_age <= config.MAX_ACCEPTABLE_FRAME_AGE_S]
        stale_count     = len(confirmed) - len(fresh_confirmed)

        target_pool = [t.detection for t in fresh_confirmed]
        using_possible_fallback = False
        if not target_pool and config.POSSIBLE_TARGET_FALLBACK_ENABLED:
            fresh_possible = [t for t in possible
                              if t.frame_age <= config.MAX_ACCEPTABLE_FRAME_AGE_S]
            target_pool = [t.detection for t in fresh_possible
                           if t.fused_confidence >= config.POSSIBLE_TARGET_MIN_CONF]
            using_possible_fallback = bool(target_pool)

        target   = self.robot.choose_target(target_pool, gripper_x, gripper_y)
        movement = self.robot.generate_movementstring(gripper_x, gripper_y)
        dx, dy   = self.robot.generate_dx(gripper_x), self.robot.generate_dy(gripper_y)
        mode     = "possible" if using_possible_fallback else "confirmed"
        print(f"[ROBOT][{mode}] {movement}: X{dx}, Y{dy}, {self.robot.generate_depthstring()}")
        self.robot.drive_hardware(gripper_x, gripper_y)

        target_id, target_center = None, None
        if target is not None:
            target_center = (target.center_x, target.center_y)
            search_list   = confirmed if not using_possible_fallback else confirmed + possible
            for t in search_list:
                d = t.detection
                if (d.x1 == target.detection.x1 and d.y1 == target.detection.y1 and
                        d.x2 == target.detection.x2 and d.y2 == target.detection.y2):
                    target_id = t.id
                    break

        self._last_movement     = movement
        self._last_target_id    = target_id
        self._last_target_center= target_center
        self._last_debug = self._make_debug(
            len(ai_dets), len(cv_dets), len(matches), len(fused),
            len(confirmed), len(possible), stale_count)
        self._last_mask = mask

        annotated = self.draw_annotations(
            frame, ai_dets, cv_dets, confirmed, possible,
            self.frame_count, gripper_x, gripper_y,
            target_id, target_center, movement, use_predicted=False,
        )
        return annotated, confirmed, self._last_debug, mask

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_servo_panel(frame: np.ndarray) -> None:
        states = servo_status.get_all()
        if not states:
            return
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        row_h, padding, col_gap = 18, 6, 8
        header    = ("ID", "Name", "Status", "Speed", "HW")
        col_texts = [(f"ID{s.id:02d}", s.name, s.status,
                      f"{s.speed}" if s.speed > 0 else "—",
                      "SIM" if s.simulated else "REAL")
                     for s in states]
        all_rows  = [header] + col_texts
        col_widths = [
            max((cv2.getTextSize(row[c], font, fs, th)[0][0] for row in all_rows)) + col_gap
            for c in range(len(header))
        ]
        panel_w = sum(col_widths) + padding * 2
        panel_h = (len(states) + 2) * row_h + padding * 2
        fh, fw  = frame.shape[:2]
        x0, y0  = fw - panel_w - 4, fh - panel_h - 4

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        hdr_y = y0 + padding + row_h
        xc    = x0 + padding
        for c, hdr in enumerate(header):
            cv2.putText(frame, hdr, (xc, hdr_y), font, fs, (200, 200, 200), th, cv2.LINE_AA)
            xc += col_widths[c]
        div_y = hdr_y + 4
        cv2.line(frame, (x0 + padding, div_y), (x0 + panel_w - padding, div_y), (80, 80, 80), 1)

        STATUS_COLORS = {
            "STOP": (160, 160, 160), "FORWARD": (0, 220, 0), "BACKWARD": (0, 140, 220),
            "LEFT": (0, 220, 0), "RIGHT": (0, 140, 220), "UP": (0, 220, 0), "DOWN": (0, 140, 220),
            "GRIP": (0, 200, 255), "GRIPPED": (0, 200, 255), "OPEN": (180, 180, 0),
            "BUSY": (0, 165, 255), "EXTENDING": (0, 220, 0),
        }
        row_y = div_y + row_h
        for s, cols in zip(states, col_texts):
            xc = x0 + padding
            sc = STATUS_COLORS.get(s.status, (220, 220, 220))
            hc = (120, 120, 120) if s.simulated else (0, 220, 100)
            for c_idx, text in enumerate(cols):
                color = sc if c_idx == 2 else hc if c_idx == 4 else (220, 220, 220)
                cv2.putText(frame, text, (xc, row_y), font, fs, color, th, cv2.LINE_AA)
                xc += col_widths[c_idx]
            row_y += row_h

    @staticmethod
    def draw_annotations(
        frame, ai_dets, cv_dets, confirmed, possible,
        frame_count, gripper_x, gripper_y,
        target_id, target_center, movement_text,
        use_predicted=False,
    ) -> np.ndarray:
        out = frame
        now = time.monotonic()

        if not use_predicted:
            for det in ai_dets:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), det.color, 1)
                cv2.putText(out, f"AI:{det.class_name}:{det.confidence:.2f}",
                            (det.x1, det.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, det.color, 1)
            for det in cv_dets:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), det.color, 1)
                cv2.putText(out, f"CV:{det.confidence:.2f}",
                            (det.x1, det.y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, det.color, 1)

        def _draw_track(track: KalmanBoxTrack, color, label_prefix, font_scale, thickness):
            det  = track.detection
            vx, vy = track.velocity
            stale  = track.frame_age > config.MAX_ACCEPTABLE_FRAME_AGE_S
            draw_color = (90, 90, 90) if stale else color

            if use_predicted:
                px1, py1, px2, py2 = track.predicted_box_now
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                w = px2 - px1; h = py2 - py1
                _draw_ghost_trail(out, pcx, pcy, vx, vy, w, h, draw_color)
                cv2.rectangle(out, (px1, py1), (px2, py2), draw_color, 1)
                _draw_velocity_arrow(out, pcx, pcy, vx, vy, draw_color)
                cv2.putText(out, f"{label_prefix}#{track.id} ~{det.confidence:.2f}",
                            (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, draw_color, 1)
            else:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), draw_color, thickness)
                cx = (det.x1 + det.x2) / 2.0; cy = (det.y1 + det.y2) / 2.0
                _draw_velocity_arrow(out, cx, cy, vx, vy, draw_color)
                age_tag = f" stale={track.frame_age:.2f}s" if stale else ""
                cv2.putText(out,
                    f"{label_prefix}#{track.id} {det.confidence:.2f}{age_tag}",
                    (det.x1, det.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, draw_color, thickness)

        for t in confirmed:
            color = (0, 165, 255) if t.id == target_id else config.COLOR_FUSED
            _draw_track(t, color, "", 0.5, 2)

        for t in possible:
            _draw_track(t, config.COLOR_POSSIBLE, "P", 0.45, 1)

        # Gripper overlay
        gripping = False
        try:
            import gripper as _gripper
            gripping = "grip" in _gripper.get_state().lower()
        except Exception:
            pass

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = RobotController.get_gripper_bbox(gripper_x, gripper_y)
        contained = False
        if target_center is not None:
            for t in confirmed + possible:
                if t.id == target_id:
                    det_for_check = t.detection
                    if use_predicted:
                        px1, py1, px2, py2 = t.predicted_box_now
                        det_for_check = Detection(px1, py1, px2, py2,
                            confidence=t.detection.confidence,
                            source=t.detection.source,
                            label=t.detection.label,
                            class_id=t.detection.class_id)
                    contained = RobotController.is_detection_fully_contained(
                        det_for_check, (bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                    break

        gc = (0, 0, 255) if gripping else (0, 255, 0) if contained else (255, 0, 255)
        cv2.rectangle(out, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), gc, 2)
        cv2.circle(out, (gripper_x, gripper_y), 8, gc, -1)
        cv2.putText(out, "GRIPPER", (gripper_x + 10, gripper_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)
        if target_center is not None:
            cv2.line(out, (gripper_x, gripper_y), target_center, (0, 165, 255), 2)

        # Legend
        legend_items = [
            (CLASS_COLORS[0], "Strawberry (targetable)"),
            (CLASS_COLORS[1], "Bad (display only)"),
            (CLASS_COLORS[2], "Leaf (display only)"),
        ]
        lx, ly = 10, frame.shape[0] - 10 - len(legend_items) * 20
        for color, label in legend_items:
            cv2.circle(out, (lx + 6, ly + 6), 5, color, -1)
            cv2.putText(out, label, (lx + 16, ly + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            ly += 20

        # HUD
        ai_badge   = "AI:ON" if is_ai_enabled() else "AI:OFF"
        cv_badge   = "CV:ON" if is_cv_enabled() else "CV:OFF"
        pred_badge = " [PRED]" if use_predicted else ""
        cv2.putText(out,
            f"Frame {frame_count} | "
            f"Hits:{len(confirmed)} | "
            f"Possible:{len(possible)} | "
            f"{ai_badge} {cv_badge}{pred_badge}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"Robot: {movement_text}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return out

    def shutdown(self) -> None:
        self._worker.stop()
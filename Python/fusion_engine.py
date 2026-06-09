import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
import servo_status
from detection import AIDetector, CVDectector, Detection, CLASS_COLORS, CLASS_NAMES, iou
from robot_controller import RobotController


INFER_SCALE               = 1
DETECT_EVERY              = 1
RECHECK_EVERY_N_DETECTIONS = 3
ZOOM_QUEUE_MAXSIZE        = 4
CLEANUP_INTERVAL          = 30
CONTAINMENT_MATCH_THRESHOLD = 0.45
MIN_TARGET_BOX_AREA       = 900   # px²
VELOCITY_ALPHA            = 0.4
MAX_PREDICT_AGE           = 0.25  # s
VELOCITY_HISTORY_LEN      = 6
ARROW_LOOKAHEAD_SEC       = 0.5
ARROW_MIN_PX              = 8
TRAIL_STEPS               = 4
TRAIL_MAX_SEC             = 0.5

# ---------------------------------------------------------------------------
# AI toggle
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_det(det: Detection, scale: float) -> Detection:
    return Detection(
        x1=int(det.x1 * scale), y1=int(det.y1 * scale),
        x2=int(det.x2 * scale), y2=int(det.y2 * scale),
        confidence=det.confidence, source=det.source,
        label=det.label, class_id=det.class_id,
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

def _is_cv_like(src: str) -> bool:
    return src.startswith("cv") or "zoomed_cv" in src

def _is_ai_like(src: str) -> bool:
    return src.startswith("ai") or "zoomed_ai" in src

# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    id: int
    detection: Detection
    seen_count: int   = 1
    missed_count: int = 0
    fused_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)

    _vx: float = field(default=0.0, init=False, repr=False)
    _vy: float = field(default=0.0, init=False, repr=False)
    _last_cx: float = field(default=0.0, init=False, repr=False)
    _last_cy: float = field(default=0.0, init=False, repr=False)
    _last_update_time: float = field(default_factory=time.time, init=False, repr=False)
    _pos_history: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        cx = (self.detection.x1 + self.detection.x2) / 2.0
        cy = (self.detection.y1 + self.detection.y2) / 2.0
        self._last_cx = cx; self._last_cy = cy
        self._last_update_time = time.time()
        self._pos_history = [(cx, cy, self._last_update_time)]

    def _push_history(self, cx, cy, now):
        self._pos_history.append((cx, cy, now))
        if len(self._pos_history) > VELOCITY_HISTORY_LEN:
            self._pos_history.pop(0)

    def _recompute_velocity(self, cx, cy, now):
        if len(self._pos_history) >= 2:
            ocx, ocy, ot = self._pos_history[0]
            dt = now - ot
            if dt > 1e-4:
                self._vx = VELOCITY_ALPHA * (cx - ocx) / dt + (1 - VELOCITY_ALPHA) * self._vx
                self._vy = VELOCITY_ALPHA * (cy - ocy) / dt + (1 - VELOCITY_ALPHA) * self._vy

    def update(self, new_det: Detection) -> None:
        now = time.time()
        cx  = (new_det.x1 + new_det.x2) / 2.0
        cy  = (new_det.y1 + new_det.y2) / 2.0
        self._push_history(cx, cy, now)
        self._recompute_velocity(cx, cy, now)
        self._last_cx = cx; self._last_cy = cy; self._last_update_time = now
        self.fused_confidence = 0.7 * new_det.confidence + 0.3 * self.fused_confidence
        self.detection = new_det
        self.seen_count += 1
        self.missed_count = 0

    def miss(self) -> None:
        self.missed_count += 1
        self.fused_confidence *= config.PERSISTENCE_DECAY

    def predicted_box(self, now=None) -> Optional[Tuple[int, int, int, int]]:
        if now is None:
            now = time.time()
        age = now - self._last_update_time
        if age > MAX_PREDICT_AGE:
            return None
        det = self.detection
        w, h = det.x2 - det.x1, det.y2 - det.y1
        pcx = self._last_cx + self._vx * age
        pcy = self._last_cy + self._vy * age
        return (int(pcx - w/2), int(pcy - h/2), int(pcx + w/2), int(pcy + h/2))

    @property
    def is_confirmed(self) -> bool:
        src = (self.detection.source or "").lower()
        req = (config.PERSISTENCE_REQUIRED_CV_ONLY
               if src.startswith("cv") or "zoomed_cv" in src
               else config.PERSISTENCE_REQUIRED)
        return self.seen_count >= req

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

# ---------------------------------------------------------------------------
# DetectionWorker
# ---------------------------------------------------------------------------

class DetectionWorker:
    def __init__(self, ai: AIDetector, cv: CVDectector) -> None:
        self.ai = ai; self.cv = cv
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._zoom_q:  queue.Queue = queue.Queue(maxsize=ZOOM_QUEUE_MAXSIZE)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._ai_dets:      List[Detection]      = []
        self._cv_dets:      List[Detection]      = []
        self._mask:         Optional[np.ndarray] = None
        self._zoom_results: List[Detection]      = []
        self._results_fresh = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection-worker")
        self._thread.start()

    def push_frame(self, frame: np.ndarray, small: np.ndarray) -> None:
        try: self._frame_q.get_nowait()
        except queue.Empty: pass
        try: self._frame_q.put_nowait(_FrameJob(frame, small))
        except queue.Full: pass

    def push_zoom(self, frame: np.ndarray, box, source: str) -> None:
        try: self._zoom_q.put_nowait(_ZoomJob(frame, box, source))
        except queue.Full: pass

    def read_frame(self):
        with self._lock:
            fresh = self._results_fresh
            self._results_fresh = False
            return self._ai_dets, self._cv_dets, self._mask, fresh

    def read_zoom(self) -> List[Detection]:
        with self._lock:
            out = self._zoom_results; self._zoom_results = []; return out

    def stop(self) -> None:
        self._stop.set(); self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try: self._process_frame(self._frame_q.get(timeout=0.05))
            except queue.Empty: pass
            try: self._process_zoom(self._zoom_q.get_nowait())
            except queue.Empty: pass

    def _process_frame(self, job: _FrameJob) -> None:
        ai_dets = self.ai.detect(job.small) if is_ai_enabled() else []
        cv_dets, mask = self.cv.detect(job.small)
        inv = 1.0 / INFER_SCALE
        ai_dets = [_scale_det(d, inv) for d in ai_dets]
        cv_dets = [_scale_det(d, inv) for d in cv_dets]
        if mask is not None:
            h, w = job.frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        with self._lock:
            self._ai_dets = ai_dets; self._cv_dets = cv_dets
            self._mask = mask; self._results_fresh = True

    def _process_zoom(self, job: _ZoomJob) -> None:
        x1, y1, x2, y2 = job.box
        h, w = job.frame.shape[:2]
        pad = max(20, (x2 - x1) // 4)
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
        roi = job.frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return
        scale  = config.ZOOM_SCALE_FACTOR
        roi_up = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))
        ai_res = self.ai.detect(roi_up, conf_threshold=config.RECHECK_AI_CONF) if is_ai_enabled() else []
        cv_res, _ = self.cv.detect(roi_up)
        sx, sy = roi.shape[1] / roi_up.shape[1], roi.shape[0] / roi_up.shape[0]

        best_det, best_conf = None, 0.0
        for det in ai_res:
            ox1, oy1 = int(rx1 + det.x1*sx), int(ry1 + det.y1*sy)
            ox2, oy2 = int(rx1 + det.x2*sx), int(ry1 + det.y2*sy)
            fused = 0.6 * det.confidence + 0.4 * self.cv.cv_score_crop(job.frame, (ox1, oy1, ox2, oy2))["total"]
            if fused > best_conf:
                best_conf = fused
                best_det  = Detection(ox1, oy1, ox2, oy2, fused, f"zoomed_{job.source}",
                                      label=det.label, class_id=det.class_id)
        for det in cv_res:
            ox1, oy1 = int(rx1 + det.x1*sx), int(ry1 + det.y1*sy)
            ox2, oy2 = int(rx1 + det.x2*sx), int(ry1 + det.y2*sy)
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_det  = Detection(ox1, oy1, ox2, oy2, det.confidence, "zoomed_cv",
                                      label="Strawberry", class_id=0)
        if best_det and best_conf >= config.RECHECK_CV_CONF:
            with self._lock:
                self._zoom_results.append(best_det)

# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _draw_velocity_arrow(img, cx, cy, vx, vy, color,
                         lookahead=ARROW_LOOKAHEAD_SEC, min_px=ARROW_MIN_PX):
    tip_x, tip_y = cx + vx * lookahead, cy + vy * lookahead
    if np.hypot(tip_x - cx, tip_y - cy) < min_px:
        return
    src, tip = (int(cx), int(cy)), (int(tip_x), int(tip_y))
    cv2.arrowedLine(img, src, tip, (255, 255, 255), thickness=4, tipLength=0.25)
    cv2.arrowedLine(img, src, tip, color,           thickness=2, tipLength=0.25)
    cv2.putText(img, f"{np.hypot(vx, vy):.0f}px/s", (tip[0]+4, tip[1]-4),
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
                      (int(gcx - gw/2), int(gcy - gh/2)),
                      (int(gcx + gw/2), int(gcy + gh/2)), color, 1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
        overlay = img.copy()
        cv2.circle(img, (int(gcx), int(gcy)), 3, color, -1)

# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    def __init__(self) -> None:
        self._worker = DetectionWorker(AIDetector(), CVDectector())
        self.robot   = RobotController()
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.recheck_counter: Dict[str, int] = defaultdict(int)
        self.frame_count   = 0
        self.detect_count  = 0
        self.last_confirmed_hits: List[TrackedObject] = []
        self.last_possible_hits:  List[TrackedObject] = []
        self._last_ai_dets:    List[Detection]      = []
        self._last_cv_dets:    List[Detection]      = []
        self._last_mask:       Optional[np.ndarray] = None
        self._last_movement    = ""
        self._last_target_id:  Optional[int]        = None
        self._last_target_center: Optional[Tuple[int, int]] = None
        self._last_debug = self._make_debug(0, 0, 0, 0, 0, 0)

    @staticmethod
    def _make_debug(ai, cv, matches, fused, confirmed, possible) -> Dict:
        return dict(ai_count=ai, cv_count=cv, matches=matches,
                    fused_count=fused, confirmed_count=confirmed, possible_count=possible)

    @staticmethod
    def _match_detections(ai_dets, cv_dets):
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
                if iou_s >= config.IOU_MATCH_THRESHOLD or cont_s >= CONTAINMENT_MATCH_THRESHOLD:
                    combined = max(iou_s, cont_s)
                    if combined > best_score:
                        best_score, best_j = combined, j
            if best_j >= 0:
                matches[i] = best_j; used_cv.add(best_j)
        return matches, [i for i in range(len(ai_dets)) if i not in matches], \
                        [j for j in range(len(cv_dets)) if j not in used_cv]

    def _request_zoom(self, frame, box, source) -> None:
        if self.detect_count % RECHECK_EVERY_N_DETECTIONS != 0:
            return
        x1, y1, x2, y2 = box
        key = f"{source}_{(x1+x2)//2//50}_{(y1+y2)//2//50}"
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

    def _classify_hits(self, tracked):
        confirmed, possible = [], []
        for obj in tracked:
            if obj.is_confirmed:
                confirmed.append(obj); continue
            src      = (obj.detection.source or "").lower()
            det_conf = float(obj.detection.confidence or 0.0)
            if _is_cv_like(src):
                min_seen, min_conf, score = config.POSSIBLE_CV_ONLY_MIN_SEEN, config.POSSIBLE_CV_ONLY_MIN_CONF, det_conf
            elif _is_ai_like(src):
                min_seen, min_conf, score = config.POSSIBLE_AI_ONLY_MIN_SEEN, config.POSSIBLE_AI_ONLY_MIN_CONF, det_conf * config.POSSIBLE_AI_CONF_WEIGHT
            else:
                min_seen, min_conf, score = config.POSSIBLE_HIT_MIN_SEEN, config.POSSIBLE_HIT_MIN_CONF, obj.fused_confidence
            if obj.seen_count >= min_seen and score >= min_conf:
                possible.append(obj)
        return confirmed, possible

    def _update_tracking(self, fused_dets):
        new_tracked, used = [], set()
        for det in fused_dets:
            if not det.is_targetable or _box_area(det) < MIN_TARGET_BOX_AREA:
                continue
            best_iou, best_id = 0.0, -1
            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used or not obj.is_active:
                    continue
                iou_s  = iou(det, obj.detection)
                cont_s = _containment(det, obj.detection)
                score  = max(iou_s, cont_s)
                if score > best_iou and (iou_s >= config.IOU_MATCH_THRESHOLD
                                         or cont_s >= CONTAINMENT_MATCH_THRESHOLD):
                    best_iou, best_id = score, obj_id
            if best_id >= 0:
                current = self.tracked_objects[best_id]
                current.update(det); new_tracked.append(current); used.add(best_id)
            else:
                obj = TrackedObject(self.next_id, det, fused_confidence=det.confidence)
                self.tracked_objects[self.next_id] = obj
                self.next_id += 1; new_tracked.append(obj)
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in used and obj.is_active:
                obj.miss()
                if obj.is_active:
                    new_tracked.append(obj)
        if self.detect_count % CLEANUP_INTERVAL == 0:
            self.tracked_objects = {oid: o for oid, o in self.tracked_objects.items() if o.is_active}
        return new_tracked

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_servo_panel(frame: np.ndarray) -> None:
        states = servo_status.get_all()
        if not states:
            return
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        row_h, padding, col_gap = 18, 6, 8

        header = ("ID", "Name", "Status", "Speed", "HW")
        col_texts = [(f"ID{s.id:02d}", s.name, s.status,
                      f"{s.speed}" if s.speed > 0 else "—",
                      "SIM" if s.simulated else "REAL")
                     for s in states]
        all_rows = [header] + col_texts
        col_widths = [
            max((cv2.getTextSize(row[c], font, fs, th)[0][0] for row in all_rows)) + col_gap
            for c in range(len(header))
        ]
        panel_w = sum(col_widths) + padding * 2
        panel_h = (len(states) + 2) * row_h + padding * 2
        fh, fw  = frame.shape[:2]
        x0, y0  = fw - panel_w - 4, fh - panel_h - 4

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (20,20,20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        hdr_y = y0 + padding + row_h
        xc = x0 + padding
        for c, hdr in enumerate(header):
            cv2.putText(frame, hdr, (xc, hdr_y), font, fs, (200,200,200), th, cv2.LINE_AA)
            xc += col_widths[c]
        div_y = hdr_y + 4
        cv2.line(frame, (x0+padding, div_y), (x0+panel_w-padding, div_y), (80,80,80), 1)

        STATUS_COLORS = {
            "STOP":(160,160,160),"FORWARD":(0,220,0),"BACKWARD":(0,140,220),
            "LEFT":(0,220,0),"RIGHT":(0,140,220),"UP":(0,220,0),"DOWN":(0,140,220),
            "GRIP":(0,200,255),"GRIPPED":(0,200,255),"OPEN":(180,180,0),
            "BUSY":(0,165,255),"EXTENDING":(0,220,0),
        }
        row_y = div_y + row_h
        for s, cols in zip(states, col_texts):
            xc = x0 + padding
            sc = STATUS_COLORS.get(s.status, (220,220,220))
            hc = (120,120,120) if s.simulated else (0,220,100)
            for c_idx, text in enumerate(cols):
                color = sc if c_idx == 2 else hc if c_idx == 4 else (220,220,220)
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
        now = time.time()

        if not use_predicted:
            for det in ai_dets:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), det.color, 1)
                cv2.putText(out, f"AI:{det.class_name}:{det.confidence:.2f}",
                            (det.x1, det.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, det.color, 1)
            for det in cv_dets:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), det.color, 1)
                cv2.putText(out, f"CV:{det.confidence:.2f}",
                            (det.x1, det.y2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, det.color, 1)

        def _draw_obj(obj, color, label_prefix, font_scale, thickness):
            det = obj.detection
            w, h = det.x2 - det.x1, det.y2 - det.y1
            if use_predicted:
                pbox = obj.predicted_box(now)
                if pbox is None:
                    return
                px1, py1, px2, py2 = pbox
                pcx, pcy = (px1+px2)/2.0, (py1+py2)/2.0
                _draw_ghost_trail(out, pcx, pcy, obj._vx, obj._vy, w, h, color)
                cv2.rectangle(out, (px1, py1), (px2, py2), color, 1)
                _draw_velocity_arrow(out, pcx, pcy, obj._vx, obj._vy, color)
                cv2.putText(out, f"{label_prefix}#{obj.id} ~{det.confidence:.2f}",
                            (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            else:
                cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
                cx, cy = (det.x1+det.x2)/2.0, (det.y1+det.y2)/2.0
                _draw_velocity_arrow(out, cx, cy, obj._vx, obj._vy, color)
                cv2.putText(out, f"{label_prefix}#{obj.id} {det.confidence:.2f}",
                            (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        for obj in confirmed:
            color = (0,165,255) if obj.id == target_id else config.COLOR_FUSED
            _draw_obj(obj, color, "", 0.5, 2)

        def _possible_score(obj):
            src = (obj.detection.source or "").lower()
            c   = float(obj.detection.confidence or 0.0)
            return c * config.POSSIBLE_AI_CONF_WEIGHT if _is_ai_like(src) else \
                   c if _is_cv_like(src) else obj.fused_confidence

        for obj in possible:
            _draw_obj(obj, config.COLOR_POSSIBLE, "P", 0.45, 1)

        # Gripper
        gripping = False
        try:
            import gripper as _gripper
            gripping = "grip" in _gripper.get_state().lower()
        except Exception:
            pass
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = RobotController.get_gripper_bbox(gripper_x, gripper_y)
        contained = False
        if target_center is not None:
            for obj in confirmed + possible:
                if obj.id == target_id:
                    det_for_check = obj.detection
                    if use_predicted:
                        pbox = obj.predicted_box(now)
                        if pbox:
                            det_for_check = Detection(*pbox,
                                confidence=obj.detection.confidence, source=obj.detection.source,
                                label=obj.detection.label, class_id=obj.detection.class_id)
                    contained = RobotController.is_detection_fully_contained(
                        det_for_check, (bbox_x1, bbox_y1, bbox_x2, bbox_y2))
                    break
        gc = (0,0,255) if gripping else (0,255,0) if contained else (255,0,255)
        cv2.rectangle(out, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), gc, 2)
        cv2.circle(out, (gripper_x, gripper_y), 8, gc, -1)
        cv2.putText(out, "GRIPPER", (gripper_x+10, gripper_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)
        if target_center is not None:
            cv2.line(out, (gripper_x, gripper_y), target_center, (0,165,255), 2)

        # Legend
        legend_items = [(CLASS_COLORS[0], "Strawberry (targetable)"),
                        (CLASS_COLORS[1], "Bad (display only)"),
                        (CLASS_COLORS[2], "Leaf (display only)")]
        lx, ly = 10, frame.shape[0] - 10 - len(legend_items) * 20
        for color, label in legend_items:
            cv2.circle(out, (lx+6, ly+6), 5, color, -1)
            cv2.putText(out, label, (lx+16, ly+11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)
            ly += 20

        # HUD
        ai_badge   = "AI:ON" if is_ai_enabled() else "AI:OFF"
        pred_badge = " [PRED]" if use_predicted else ""
        cv2.putText(out,
            f"Frame {frame_count} | Hits:{len(confirmed)} | Possible:{len(possible)} | {ai_badge}{pred_badge}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(out, f"Robot: {movement_text}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        # FusionEngine._draw_servo_panel(out) #(servo debug ui (uitgeschakeld)
        return out

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        self.frame_count += 1
        small = cv2.resize(frame, (0,0), fx=INFER_SCALE, fy=INFER_SCALE,
                           interpolation=cv2.INTER_LINEAR)
        self._worker.push_frame(frame, small)
        gripper_x = frame.shape[1] // 2
        gripper_y = frame.shape[0] // 2

        ai_dets, cv_dets, mask, fresh = self._worker.read_frame()
        if not fresh:
            annotated = self.draw_annotations(
                frame.copy(), self._last_ai_dets, self._last_cv_dets,
                self.last_confirmed_hits, self.last_possible_hits,
                self.frame_count, gripper_x, gripper_y,
                self._last_target_id, self._last_target_center,
                self._last_movement, use_predicted=True,
            )
            return annotated, self.last_confirmed_hits, self._last_debug, self._last_mask

        self.detect_count += 1
        zoom_dets = self._worker.read_zoom()
        self._last_ai_dets = ai_dets
        self._last_cv_dets = cv_dets

        ai_targetable = [d for d in ai_dets if d.is_targetable]
        if is_ai_enabled():
            matches, unmatched_ai, unmatched_cv = self._match_detections(ai_targetable, cv_dets)
        else:
            matches, unmatched_ai, unmatched_cv = {}, [], list(range(len(cv_dets)))

        fused: List[Detection] = [d for d in zoom_dets if d.is_targetable]
        for ai_idx, cv_idx in matches.items():
            maybe = self._fuse_decision(ai_targetable[ai_idx], cv_dets[cv_idx], frame)
            if maybe: fused.append(maybe)
        for ai_idx in unmatched_ai:
            det = ai_targetable[ai_idx]
            if det.confidence < config.LOW_AI_CONFIDENCE:
                self._request_zoom(frame, (det.x1, det.y1, det.x2, det.y2), "ai_low")
            else:
                maybe = self._fuse_decision(det, None, frame)
                if maybe: fused.append(maybe)
        for cv_idx in unmatched_cv:
            maybe = self._fuse_decision(None, cv_dets[cv_idx], frame)
            if maybe: fused.append(maybe)

        tracked = self._update_tracking(fused)
        confirmed, possible = self._classify_hits(tracked)
        self.last_confirmed_hits = confirmed
        self.last_possible_hits  = possible

        target_pool = [o.detection for o in confirmed]
        using_possible_fallback = False
        if not target_pool and config.POSSIBLE_TARGET_FALLBACK_ENABLED:
            def _pscore(p):
                src, c = (p.detection.source or "").lower(), float(p.detection.confidence or 0.0)
                return c if _is_cv_like(src) else c * config.POSSIBLE_AI_CONF_WEIGHT if _is_ai_like(src) else p.fused_confidence
            target_pool = [o.detection for o in possible if _pscore(o) >= config.POSSIBLE_TARGET_MIN_CONF]
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
            for obj in search_list:
                if (obj.detection.x1 == target.detection.x1 and
                        obj.detection.y1 == target.detection.y1 and
                        obj.detection.x2 == target.detection.x2 and
                        obj.detection.y2 == target.detection.y2):
                    target_id = obj.id; break

        self._last_movement = movement
        self._last_target_id = target_id
        self._last_target_center = target_center
        self._last_debug = self._make_debug(
            len(ai_dets), len(cv_dets), len(matches), len(fused), len(confirmed), len(possible))
        self._last_mask = mask

        annotated = self.draw_annotations(
            frame, ai_dets, cv_dets, confirmed, possible,
            self.frame_count, gripper_x, gripper_y,
            target_id, target_center, movement, use_predicted=False,
        )
        return annotated, confirmed, self._last_debug, mask

    def shutdown(self) -> None:
        self._worker.stop()
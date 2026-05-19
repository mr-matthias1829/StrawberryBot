"""Cascade fusion pipeline — optimised for Raspberry Pi 5 @ 15-20 fps.

Architecture
────────────
Main thread:   capture → display only.  Never touches a detector.
Worker thread: AI + CV detection + zoom rechecks, all sequential, no blocking.

The main thread pushes frames; the worker processes them and stores results.
Between worker cycles the main thread re-stamps last known boxes on the live
frame — bounding boxes feel smooth even when inference is slower.
"""

import cv2
import numpy as np
import threading
import queue
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

import config
from detection import AIDetector, CVDectector, Detection, iou
from robot_controller import RobotController


# ─────────────────────────────────────────────────────────────────────────────
#  Tuning knobs
# ─────────────────────────────────────────────────────────────────────────────

# Detectors run on a downscaled frame.  0.5 → 640×360 instead of 1280×720.
INFER_SCALE: float = 0.5

# Only re-run the fusion pipeline every N display frames.
DETECT_EVERY: int = 2

# Zoom rechecks fire at most once every N *detection* cycles per location.
RECHECK_EVERY_N_DETECTIONS: int = 3

# Max pending zoom jobs.  If the queue fills up, new jobs are dropped.
ZOOM_QUEUE_MAXSIZE: int = 4

CLEANUP_INTERVAL: int = 30

DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720


# ─────────────────────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    id: int
    detection: Detection
    seen_count: int = 1
    missed_count: int = 0
    fused_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)

    def update(self, new_det: Detection, iou_score: float) -> None:
        self.fused_confidence = 0.7 * new_det.confidence + 0.3 * self.fused_confidence
        self.detection = new_det
        self.seen_count += 1
        self.missed_count = 0

    def miss(self) -> None:
        self.missed_count += 1
        self.fused_confidence *= config.PERSISTENCE_DECAY

    @property
    def is_confirmed(self) -> bool:
        src = (self.detection.source or "").lower()
        required = (
            config.PERSISTENCE_REQUIRED_CV_ONLY
            if src.startswith("cv") or "zoomed_cv" in src
            else config.PERSISTENCE_REQUIRED
        )
        return self.seen_count >= required

    @property
    def is_active(self) -> bool:
        return self.missed_count < config.PERSISTENCE_REQUIRED


def _scale_det(det: Detection, scale: float) -> Detection:
    return Detection(
        x1=int(det.x1 * scale), y1=int(det.y1 * scale),
        x2=int(det.x2 * scale), y2=int(det.y2 * scale),
        confidence=det.confidence, source=det.source,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Background worker  (owns ALL detector calls)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FrameJob:
    frame: np.ndarray
    small: np.ndarray

@dataclass
class _ZoomJob:
    frame: np.ndarray
    box: Tuple[int, int, int, int]
    source: str


class DetectionWorker:
    """Single background thread that owns every detector call.

    Two input queues (size=1 for frames, bounded for zoom jobs) mean the
    main thread never blocks — it just drops work if the worker is busy.

    Results are read via read_frame() and read_zoom() — both non-blocking.
    """

    def __init__(self, ai: AIDetector, cv: CVDectector) -> None:
        self.ai = ai
        self.cv = cv

        # Frame queue: size 1 — always process the freshest frame.
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        # Zoom queue: bounded — drop if saturated rather than queue up stale work.
        self._zoom_q: queue.Queue = queue.Queue(maxsize=ZOOM_QUEUE_MAXSIZE)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Latest outputs
        self._ai_dets: List[Detection] = []
        self._cv_dets: List[Detection] = []
        self._mask: Optional[np.ndarray] = None
        self._zoom_results: List[Detection] = []   # accumulates between reads

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── public API (main thread) ──────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray, small: np.ndarray) -> None:
        """Drop a frame for detection.  Discards the previous pending frame."""
        try:
            self._frame_q.get_nowait()   # evict stale frame if present
        except queue.Empty:
            pass
        try:
            self._frame_q.put_nowait(_FrameJob(frame, small))
        except queue.Full:
            pass

    def push_zoom(self, frame: np.ndarray, box: Tuple[int,int,int,int], source: str) -> None:
        """Queue a zoom recheck.  Silently dropped if queue is full."""
        try:
            self._zoom_q.put_nowait(_ZoomJob(frame, box, source))
        except queue.Full:
            pass

    def read_frame(self) -> Tuple[List[Detection], List[Detection], Optional[np.ndarray]]:
        """Non-blocking: returns latest finished frame-detection results."""
        with self._lock:
            return self._ai_dets, self._cv_dets, self._mask

    def read_zoom(self) -> List[Detection]:
        """Non-blocking: drains and returns all finished zoom results."""
        with self._lock:
            results = self._zoom_results
            self._zoom_results = []
            return results

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    # ── worker loop ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            # --- Frame detection (blocking wait, 50ms timeout) ---------------
            try:
                job = self._frame_q.get(timeout=0.05)
                self._process_frame(job)
            except queue.Empty:
                pass

            # --- Drain zoom queue (process one per loop iteration) -----------
            try:
                zjob = self._zoom_q.get_nowait()
                result = self._process_zoom(zjob)
                if result:
                    with self._lock:
                        self._zoom_results.append(result)
            except queue.Empty:
                pass

    def _process_frame(self, job: _FrameJob) -> None:
        ai_dets = self.ai.detect(job.small)
        cv_dets, mask = self.cv.detect(job.small)

        inv = 1.0 / INFER_SCALE
        ai_dets = [_scale_det(d, inv) for d in ai_dets]
        cv_dets  = [_scale_det(d, inv) for d in cv_dets]

        if mask is not None:
            h, w = job.frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        with self._lock:
            self._ai_dets = ai_dets
            self._cv_dets = cv_dets
            self._mask = mask

    def _process_zoom(self, job: _ZoomJob) -> Optional[Detection]:
        x1, y1, x2, y2 = job.box
        frame = job.frame
        h, w = frame.shape[:2]

        pad  = max(20, (x2 - x1) // 4)
        rx1  = max(0, x1 - pad);  ry1 = max(0, y1 - pad)
        rx2  = min(w, x2 + pad);  ry2 = min(h, y2 + pad)

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        scale  = config.ZOOM_SCALE_FACTOR
        roi_up = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))

        ai_results = self.ai.detect(roi_up, conf_threshold=config.RECHECK_AI_CONF)
        cv_results, _ = self.cv.detect(roi_up)

        sx = roi.shape[1] / roi_up.shape[1]
        sy = roi.shape[0] / roi_up.shape[0]

        best_det: Optional[Detection] = None
        best_conf = 0.0

        for det in ai_results:
            ox1 = int(rx1 + det.x1 * sx);  oy1 = int(ry1 + det.y1 * sy)
            ox2 = int(rx1 + det.x2 * sx);  oy2 = int(ry1 + det.y2 * sy)
            scores = self.cv.cv_score_crop(frame, (ox1, oy1, ox2, oy2))
            fused  = 0.6 * det.confidence + 0.4 * scores["total"]
            if fused > best_conf:
                best_conf = fused
                best_det  = Detection(x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                                      confidence=fused, source=f"zoomed_{job.source}")

        for det in cv_results:
            ox1 = int(rx1 + det.x1 * sx);  oy1 = int(ry1 + det.y1 * sy)
            ox2 = int(rx1 + det.x2 * sx);  oy2 = int(ry1 + det.y2 * sy)
            if det.confidence > best_conf:
                best_conf = det.confidence
                best_det  = Detection(x1=ox1, y1=oy1, x2=ox2, y2=oy2,
                                      confidence=det.confidence, source="zoomed_cv")

        return best_det if best_conf >= config.RECHECK_CV_CONF else None


# ─────────────────────────────────────────────────────────────────────────────
#  Fusion Engine  (main-thread logic only — no detector calls here)
# ─────────────────────────────────────────────────────────────────────────────

class FusionEngine:

    def __init__(self) -> None:
        ai_det = AIDetector()
        cv_det = CVDectector()

        self._worker = DetectionWorker(ai_det, cv_det)

        # ─────────────────────────────────────────
        # ROBOT CONTROLLER
        # ─────────────────────────────────────────
        self.robot = RobotController()

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.recheck_counter: Dict[str, int] = defaultdict(int)

        self.frame_count = 0
        self.detect_count = 0

        self.last_confirmed_hits: List[TrackedObject] = []
        self.last_possible_hits: List[TrackedObject] = []

        self._last_annotated: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None

        self._last_debug: Dict = {
            "ai_count": 0,
            "cv_count": 0,
            "matches": 0,
            "fused_count": 0,
            "confirmed_count": 0,
            "possible_count": 0,
        }

    # ─────────────────────────────────────────────
    # SOURCE HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _is_cv_like(src: str) -> bool:
        return src.startswith("cv") or "zoomed_cv" in src

    @staticmethod
    def _is_ai_like(src: str) -> bool:
        return src.startswith("ai") or "zoomed_ai" in src

    # ─────────────────────────────────────────────
    # DETECTION MATCHING
    # ─────────────────────────────────────────────

    @staticmethod
    def _match_detections(
        ai_dets: List[Detection],
        cv_dets: List[Detection]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:

        matches: Dict[int, int] = {}
        used_cv: set = set()

        for i, ai in enumerate(ai_dets):

            best_iou = 0.0
            best_j = -1

            for j, cv in enumerate(cv_dets):

                if j in used_cv:
                    continue

                score = iou(ai, cv)

                if score > best_iou and score >= config.IOU_MATCH_THRESHOLD:
                    best_iou = score
                    best_j = j

            if best_j >= 0:
                matches[i] = best_j
                used_cv.add(best_j)

        unmatched_ai = [
            i for i in range(len(ai_dets))
            if i not in matches
        ]

        unmatched_cv = [
            j for j in range(len(cv_dets))
            if j not in used_cv
        ]

        return matches, unmatched_ai, unmatched_cv

    # ─────────────────────────────────────────────
    # ZOOM RECHECK
    # ─────────────────────────────────────────────

    def _request_zoom(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        source: str
    ) -> None:

        if self.detect_count % RECHECK_EVERY_N_DETECTIONS != 0:
            return

        x1, y1, x2, y2 = box

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        key = f"{source}_{cx // 50}_{cy // 50}"

        if self.recheck_counter[key] >= config.MAX_RECHECKS:
            return

        self.recheck_counter[key] += 1

        self._worker.push_zoom(frame, box, source)

    # ─────────────────────────────────────────────
    # FUSION DECISION
    # ─────────────────────────────────────────────

    def _fuse_decision(
        self,
        ai_det: Optional[Detection],
        cv_det: Optional[Detection],
        frame: np.ndarray,
    ) -> Optional[Detection]:

        if ai_det and cv_det:

            fused = (
                config.YOLO_FUSION_WEIGHT * ai_det.confidence
                + config.CV_FUSION_WEIGHT * cv_det.confidence
            )

            return Detection(
                x1=ai_det.x1,
                y1=ai_det.y1,
                x2=ai_det.x2,
                y2=ai_det.y2,
                confidence=fused,
                source="fused"
            )

        if ai_det:

            if ai_det.confidence > config.HIGH_AI_CONFIDENCE:

                return Detection(
                    x1=ai_det.x1,
                    y1=ai_det.y1,
                    x2=ai_det.x2,
                    y2=ai_det.y2,
                    confidence=ai_det.confidence,
                    source="ai_high"
                )

            self._request_zoom(
                frame,
                (ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2),
                "ai"
            )

            return None

        if cv_det:

            if cv_det.confidence >= config.CV_DIRECT_ACCEPT_THRESHOLD:
                return cv_det

            self._request_zoom(
                frame,
                (cv_det.x1, cv_det.y1, cv_det.x2, cv_det.y2),
                "cv"
            )

            return None

        return None

    # ─────────────────────────────────────────────
    # CLASSIFICATION
    # ─────────────────────────────────────────────

    def _classify_hits(
        self,
        tracked: List[TrackedObject]
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:

        confirmed: List[TrackedObject] = []
        possible: List[TrackedObject] = []

        for obj in tracked:

            if obj.is_confirmed:
                confirmed.append(obj)
                continue

            src = (obj.detection.source or "").lower()
            score = obj.fused_confidence

            if self._is_cv_like(src):

                min_seen = config.POSSIBLE_CV_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_CV_ONLY_MIN_CONF

            elif self._is_ai_like(src):

                min_seen = config.POSSIBLE_AI_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_AI_ONLY_MIN_CONF

                score *= config.POSSIBLE_AI_CONF_WEIGHT

            else:

                min_seen = config.POSSIBLE_HIT_MIN_SEEN
                min_conf = config.POSSIBLE_HIT_MIN_CONF

            if obj.seen_count >= min_seen and score >= min_conf:
                possible.append(obj)

        return confirmed, possible

    # ─────────────────────────────────────────────
    # TRACKING
    # ─────────────────────────────────────────────

    def _update_tracking(
        self,
        fused_dets: List[Detection]
    ) -> List[TrackedObject]:

        new_tracked: List[TrackedObject] = []
        used: set = set()

        for det in fused_dets:

            best_iou = 0.0
            best_id = -1

            for obj_id, obj in self.tracked_objects.items():

                if obj_id in used or not obj.is_active:
                    continue

                score = iou(det, obj.detection)

                if score > best_iou and score >= config.IOU_MATCH_THRESHOLD:
                    best_iou = score
                    best_id = obj_id

            if best_id >= 0:

                self.tracked_objects[best_id].update(det, best_iou)

                new_tracked.append(
                    self.tracked_objects[best_id]
                )

                used.add(best_id)

            else:

                obj = TrackedObject(
                    id=self.next_id,
                    detection=det,
                    fused_confidence=det.confidence
                )

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
                oid: o
                for oid, o in self.tracked_objects.items()
                if o.is_active
            }

        return new_tracked

    # ─────────────────────────────────────────────
    # DRAWING
    # ─────────────────────────────────────────────

    @staticmethod
    def _draw_annotations(
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
    ) -> np.ndarray:

        out = frame.copy()

        # ─────────────────────────────────────────
        # AI DETS
        # ─────────────────────────────────────────

        for det in ai_dets:

            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                config.COLOR_AI,
                1
            )

            cv2.putText(
                out,
                f"AI:{det.confidence:.2f}",
                (det.x1, det.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                config.COLOR_AI,
                1
            )

        # ─────────────────────────────────────────
        # CV DETS
        # ─────────────────────────────────────────

        for det in cv_dets:

            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                config.COLOR_CV,
                1
            )

            cv2.putText(
                out,
                f"CV:{det.confidence:.2f}",
                (det.x1, det.y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                config.COLOR_CV,
                1
            )

        # ─────────────────────────────────────────
        # CONFIRMED
        # ─────────────────────────────────────────

        for obj in confirmed:

            det = obj.detection

            color = config.COLOR_FUSED

            # TARGET = ORANGE
            if obj.id == target_id:
                color = (0, 165, 255)

            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                color,
                2
            )

            cv2.putText(
                out,
                f"#{obj.id} {det.confidence:.2f}",
                (det.x1, det.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # ─────────────────────────────────────────
        # POSSIBLE
        # ─────────────────────────────────────────

        for obj in possible:

            det = obj.detection

            cv2.rectangle(
                out,
                (det.x1, det.y1),
                (det.x2, det.y2),
                config.COLOR_POSSIBLE,
                1
            )

            cv2.putText(
                out,
                f"P#{obj.id} {obj.fused_confidence:.2f}",
                (det.x1, det.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                config.COLOR_POSSIBLE,
                1
            )

        # ─────────────────────────────────────────
        # GRIPPER POINT
        # ─────────────────────────────────────────

        cv2.circle(
            out,
            (gripper_x, gripper_y),
            8,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            out,
            "GRIPPER",
            (gripper_x + 10, gripper_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        # ─────────────────────────────────────────
        # TARGET LINE
        # ─────────────────────────────────────────

        if target_center is not None:

            cv2.line(
                out,
                (gripper_x, gripper_y),
                target_center,
                (0, 165, 255),
                2
            )

        # ─────────────────────────────────────────
        # STATUS TEXT
        # ─────────────────────────────────────────

        cv2.putText(
            out,
            f"Frame {frame_count} | Hits: {len(confirmed)} | Possible: {len(possible)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            out,
            f"Robot: {movement_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2
        )

        return out

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[TrackedObject], Dict, Optional[np.ndarray]]:

        self.frame_count += 1

        small = cv2.resize(
            frame,
            (0, 0),
            fx=INFER_SCALE,
            fy=INFER_SCALE,
            interpolation=cv2.INTER_LINEAR
        )

        self._worker.push_frame(frame, small)

        if (
            self.frame_count % DETECT_EVERY != 0
            and self._last_annotated is not None
        ):

            return (
                self._last_annotated,
                self.last_confirmed_hits,
                self._last_debug,
                self._last_mask
            )

        # ─────────────────────────────────────────
        # DETECTION
        # ─────────────────────────────────────────

        self.detect_count += 1

        ai_dets, cv_dets, mask = self._worker.read_frame()

        zoom_dets = self._worker.read_zoom()

        matches, unmatched_ai, unmatched_cv = self._match_detections(
            ai_dets,
            cv_dets
        )

        fused: List[Detection] = list(zoom_dets)

        for ai_idx, cv_idx in matches.items():

            r = self._fuse_decision(
                ai_dets[ai_idx],
                cv_dets[cv_idx],
                frame
            )

            if r:
                fused.append(r)

        for ai_idx in unmatched_ai:

            det = ai_dets[ai_idx]

            if det.confidence < config.LOW_AI_CONFIDENCE:

                self._request_zoom(
                    frame,
                    (det.x1, det.y1, det.x2, det.y2),
                    "ai_low"
                )

            else:

                r = self._fuse_decision(det, None, frame)

                if r:
                    fused.append(r)

        for cv_idx in unmatched_cv:

            r = self._fuse_decision(
                None,
                cv_dets[cv_idx],
                frame
            )

            if r:
                fused.append(r)

        tracked = self._update_tracking(fused)

        confirmed, possible = self._classify_hits(tracked)

        self.last_confirmed_hits = confirmed
        self.last_possible_hits = possible

        # ─────────────────────────────────────────
        # ROBOT TARGETING
        # ─────────────────────────────────────────

        gripper_x = frame.shape[1] // 2
        gripper_y = frame.shape[0] // 2

        confirmed_dets = [
            obj.detection
            for obj in confirmed
        ]

        target = self.robot.choose_target(
            confirmed_dets,
            gripper_x,
            gripper_y
        )

        movement = self.robot.generate_movement(
            gripper_x,
            gripper_y
        )

        print(f"[ROBOT] {movement}")

        target_id = None
        target_center = None

        if target is not None:

            target_center = (
                target.center_x,
                target.center_y
            )

            for obj in confirmed:

                if obj.detection == target.detection:
                    target_id = obj.id
                    break

        # ─────────────────────────────────────────
        # DEBUG
        # ─────────────────────────────────────────

        self._last_debug = {
            "ai_count": len(ai_dets),
            "cv_count": len(cv_dets),
            "matches": len(matches),
            "fused_count": len(fused),
            "confirmed_count": len(confirmed),
            "possible_count": len(possible),
        }

        self._last_mask = mask

        # ─────────────────────────────────────────
        # DRAW
        # ─────────────────────────────────────────

        self._last_annotated = self._draw_annotations(
            frame,
            ai_dets,
            cv_dets,
            confirmed,
            possible,
            self.frame_count,
            gripper_x,
            gripper_y,
            target_id,
            target_center,
            movement
        )

        return (
            self._last_annotated,
            confirmed,
            self._last_debug,
            mask
        )

    def shutdown(self) -> None:
        self._worker.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(rtsp_url: str = "rtsp://admin:admin@192.168.42.1:554/live") -> None:
    print(f"Connecting to reCamera stream: {rtsp_url}")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    connected = False
    if cap.isOpened():
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                connected = True
                break
            time.sleep(0.1)

    if not connected:
        print("⚠️  reCamera not available — falling back to laptop camera.")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("❌ No camera available at all.")
            return
        print("✅ Laptop camera connected.")
    else:
        print("✅ reCamera stream connected.")

    fusion = FusionEngine()
    print("\nPipeline active. Press 'q' to quit, 'd' to toggle debug mask.")
    print(f"Inference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")

    show_mask = config.SHOW_DEBUG_WINDOWS
    fps_timer = time.perf_counter()
    fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Dropped frame, retrying...")
                continue

            annotated, confirmed, debug, mask = fusion.process_frame(frame)

            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                print(
                    f"FPS: {fps_count:2d} | "
                    f"AI: {debug['ai_count']} CV: {debug['cv_count']} "
                    f"Fused: {debug['fused_count']} "
                    f"Hits: {debug['confirmed_count']} "
                    f"Possible: {debug['possible_count']}"
                )
                fps_count = 0
                fps_timer = now

            display = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("Strawberry Detection", display)
            if show_mask and mask is not None:
                cv2.imshow("CV Mask", cv2.resize(mask, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("CV Mask")
    finally:
        fusion.shutdown()
        cap.release()
        cv2.destroyAllWindows()


def run_image(image_path: str = None) -> None:
    import os
    if image_path is None:
        image_path = os.path.join(
            os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg"
        )

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    fusion = FusionEngine()
    fusion.process_frame(frame)
    time.sleep(0.5)                    # let background worker finish
    annotated, confirmed, debug, mask = fusion.process_frame(frame)
    fusion.shutdown()

    print(f"\nResults for {image_path}:")
    print(f"  AI: {debug['ai_count']} | CV: {debug['cv_count']} | "
          f"Fused: {debug['fused_count']} | Hits: {debug['confirmed_count']} "
          f"| Possible: {debug['possible_count']}")

    for obj in confirmed:
        det = obj.detection
        print(f"  Berry {obj.id}: conf={det.confidence:.3f}, "
              f"seen={obj.seen_count} frames, source={det.source}")

    if fusion.last_possible_hits:
        print("  Possible hits:")
        for obj in fusion.last_possible_hits:
            det = obj.detection
            print(f"    P{obj.id}: conf={obj.fused_confidence:.3f}, "
                  f"seen={obj.seen_count} frames, source={det.source}")

    cv2.imshow("Result", annotated)
    if mask is not None:
        cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        run_image(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://admin:admin@192.168.42.1:554/live"
        run_webcam(url)
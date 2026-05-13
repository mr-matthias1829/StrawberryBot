"""Main entry point with cascade fusion pipeline."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

import config
from detection import AIDetector, CVDectector, Detection, iou


@dataclass
class TrackedObject:
    """Object tracked across frames with temporal persistence."""
    id: int
    detection: Detection
    seen_count: int = 1
    missed_count: int = 0
    fused_confidence: float = 0.0
    first_seen: float = field(default_factory=time.time)

    def update(self, new_det: Detection, iou_score: float):
        """Update tracked object with new detection."""
        # Exponential moving average for confidence
        self.fused_confidence = 0.7 * new_det.confidence + 0.3 * self.fused_confidence
        self.detection = new_det
        self.seen_count += 1
        self.missed_count = 0

    def miss(self):
        """Mark frame where object wasn't seen."""
        self.missed_count += 1
        # Decay confidence when missing
        self.fused_confidence *= config.PERSISTENCE_DECAY

    @property
    def is_confirmed(self) -> bool:
        """Use stricter persistence for CV-only tracks than fused/AI tracks."""
        src = (self.detection.source or "").lower()
        required = (
            config.PERSISTENCE_REQUIRED_CV_ONLY
            if src.startswith("cv") or "zoomed_cv" in src
            else config.PERSISTENCE_REQUIRED
        )
        return self.seen_count >= required

    @property
    def is_active(self) -> bool:
        """Object still considered active."""
        return self.missed_count < config.PERSISTENCE_REQUIRED


class FusionEngine:
    """Cascade fusion: AI + CV with refinement and temporal memory."""

    def __init__(self):
        self.ai_detector = AIDetector()
        self.cv_detector = CVDectector()
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.recheck_counter: Dict[str, int] = defaultdict(int)
        self.frame_count = 0
        self.last_confirmed_hits: List[TrackedObject] = []
        self.last_possible_hits: List[TrackedObject] = []

    def _classify_hits(self, tracked_objects: List[TrackedObject]) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Split tracked objects into confirmed hits and possible hits."""
        def _is_cv_like(source: str) -> bool:
            return source.startswith("cv") or "zoomed_cv" in source

        def _is_ai_like(source: str) -> bool:
            return source.startswith("ai") or "zoomed_ai" in source

        confirmed = [obj for obj in tracked_objects if obj.is_confirmed]
        possible = []
        for obj in tracked_objects:
            if obj.is_confirmed:
                continue

            src = (obj.detection.source or "").lower()
            score = obj.fused_confidence

            # Disagreement policy: CV-only gets a looser possible gate;
            # AI-only is down-weighted to suppress face/false-positive drift.
            if _is_cv_like(src):
                min_seen = config.POSSIBLE_CV_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_CV_ONLY_MIN_CONF
            elif _is_ai_like(src):
                min_seen = config.POSSIBLE_AI_ONLY_MIN_SEEN
                min_conf = config.POSSIBLE_AI_ONLY_MIN_CONF
                score *= config.POSSIBLE_AI_CONF_WEIGHT
            else:
                min_seen = config.POSSIBLE_HIT_MIN_SEEN
                min_conf = config.POSSIBLE_HIT_MIN_CONF

            if obj.seen_count < min_seen:
                continue
            if score < min_conf:
                continue
            possible.append(obj)
        return confirmed, possible

    def _match_detections(self, ai_dets: List[Detection], cv_dets: List[Detection]) -> Tuple[
        Dict[int, int], List[int], List[int]
    ]:
        """Match AI and CV detections using IoU."""
        matches = {}
        used_cv = set()

        for i, ai in enumerate(ai_dets):
            best_iou = 0
            best_j = -1
            for j, cv in enumerate(cv_dets):
                if j in used_cv:
                    continue
                iou_score = iou(ai, cv)
                if iou_score > best_iou and iou_score >= config.IOU_MATCH_THRESHOLD:
                    best_iou = iou_score
                    best_j = j
            if best_j >= 0:
                matches[i] = best_j
                used_cv.add(best_j)

        unmatched_ai = [i for i in range(len(ai_dets)) if i not in matches]
        unmatched_cv = [j for j in range(len(cv_dets)) if j not in used_cv]

        return matches, unmatched_ai, unmatched_cv

    def _zoom_recheck(self, frame: np.ndarray, box: Tuple[int, int, int, int],
                      source: str = "ai") -> Optional[Detection]:
        """Zoom into ROI and re-run detectors for refinement."""
        x1, y1, x2, y2 = box

        # Prevent infinite recursion with location-based key
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        key = f"{source}_{cx//50}_{cy//50}"
        if self.recheck_counter[key] >= config.MAX_RECHECKS:
            return None

        self.recheck_counter[key] += 1

        # Expand ROI slightly for context
        h, w = frame.shape[:2]
        pad = max(20, (x2 - x1) // 4)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Upscale
        scale = config.ZOOM_SCALE_FACTOR
        new_w, new_h = int(roi.shape[1] * scale), int(roi.shape[0] * scale)
        roi_upscaled = cv2.resize(roi, (new_w, new_h))

        # Run both detectors on upscaled ROI
        ai_results = self.ai_detector.detect(roi_upscaled, conf_threshold=config.RECHECK_AI_CONF)
        cv_results, _ = self.cv_detector.detect(roi_upscaled)

        # Scale coordinates back
        scale_x = roi.shape[1] / new_w
        scale_y = roi.shape[0] / new_h

        best_det = None
        best_conf = 0

        # Check AI results first
        for det in ai_results:
            orig_x1 = int(x1 + det.x1 * scale_x)
            orig_y1 = int(y1 + det.y1 * scale_y)
            orig_x2 = int(x1 + det.x2 * scale_x)
            orig_y2 = int(y1 + det.y2 * scale_y)

            # Re-score with CV on original frame coordinates
            scores = self.cv_detector.cv_score_crop(frame, (orig_x1, orig_y1, orig_x2, orig_y2))
            fused_conf = 0.6 * det.confidence + 0.4 * scores["total"]

            if fused_conf > best_conf:
                best_conf = fused_conf
                best_det = Detection(
                    x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2,
                    confidence=fused_conf, source=f"zoomed_{source}"
                )

        # Also check CV-only detections
        for det in cv_results:
            orig_x1 = int(x1 + det.x1 * scale_x)
            orig_y1 = int(y1 + det.y1 * scale_y)
            orig_x2 = int(x1 + det.x2 * scale_x)
            orig_y2 = int(y1 + det.y2 * scale_y)

            if det.confidence > best_conf:
                best_conf = det.confidence
                best_det = Detection(
                    x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2,
                    confidence=det.confidence, source=f"zoomed_cv"
                )

        return best_det if best_conf >= config.RECHECK_CV_CONF else None

    def _fuse_decision(self, ai_det: Optional[Detection], cv_det: Optional[Detection],
                       frame: np.ndarray) -> Optional[Detection]:
        """
        Cascade decision logic from design doc.
        """
        # Case A: both agree → fusion
        if ai_det and cv_det:
            fused_conf = (config.YOLO_FUSION_WEIGHT * ai_det.confidence +
                         config.CV_FUSION_WEIGHT * cv_det.confidence)
            return Detection(
                x1=ai_det.x1, y1=ai_det.y1, x2=ai_det.x2, y2=ai_det.y2,
                confidence=fused_conf, source="fused"
            )

        # Case B: AI only, CV no
        if ai_det and not cv_det:
            if ai_det.confidence > config.HIGH_AI_CONFIDENCE:
                return Detection(
                    x1=ai_det.x1, y1=ai_det.y1, x2=ai_det.x2, y2=ai_det.y2,
                    confidence=ai_det.confidence, source="ai_high"
                )
            else:
                # Zoom recheck
                box = (ai_det.x1, ai_det.y1, ai_det.x2, ai_det.y2)
                return self._zoom_recheck(frame, box, source="ai")

        # Case C: CV only, AI no
        if cv_det and not ai_det:
            # Score with CV first - if already high confidence, trust it
            if cv_det.confidence >= config.CV_DIRECT_ACCEPT_THRESHOLD:
                return cv_det
            # Otherwise zoom recheck
            box = (cv_det.x1, cv_det.y1, cv_det.x2, cv_det.y2)
            return self._zoom_recheck(frame, box, source="cv")

        return None

    def _update_tracking(self, fused_detections: List[Detection]) -> List[TrackedObject]:
        """Update temporal tracking with new detections."""
        new_tracked = []
        used_tracked = set()

        # Match existing tracked objects with new detections
        for det in fused_detections:
            best_iou = 0
            best_id = -1
            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used_tracked or not obj.is_active:
                    continue
                iou_score = iou(det, obj.detection)
                if iou_score > best_iou and iou_score >= config.IOU_MATCH_THRESHOLD:
                    best_iou = iou_score
                    best_id = obj_id

            if best_id >= 0:
                self.tracked_objects[best_id].update(det, best_iou)
                new_tracked.append(self.tracked_objects[best_id])
                used_tracked.add(best_id)
            else:
                obj = TrackedObject(id=self.next_id, detection=det, fused_confidence=det.confidence)
                self.tracked_objects[self.next_id] = obj
                self.next_id += 1
                new_tracked.append(obj)

        # Mark missed objects
        for obj_id, obj in self.tracked_objects.items():
            if obj_id not in used_tracked and obj.is_active:
                obj.miss()
                if obj.is_active:
                    new_tracked.append(obj)

        # Clean up dead objects
        self.tracked_objects = {obj_id: obj for obj_id, obj in self.tracked_objects.items()
                                if obj.is_active}

        return new_tracked

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TrackedObject], Dict, np.ndarray]:
        """Process single frame through full cascade pipeline."""
        self.frame_count += 1

        # Stage 1: Parallel detection
        ai_dets = self.ai_detector.detect(frame)
        cv_dets, mask = self.cv_detector.detect(frame)

        # Stage 2: Match detections
        matches, unmatched_ai, unmatched_cv = self._match_detections(ai_dets, cv_dets)

        # Stage 3: Decision fusion
        fused_detections = []
        debug_info = {
            "ai_count": len(ai_dets),
            "cv_count": len(cv_dets),
            "matches": len(matches)
        }

        # Matched pairs
        for ai_idx, cv_idx in matches.items():
            result = self._fuse_decision(ai_dets[ai_idx], cv_dets[cv_idx], frame)
            if result:
                fused_detections.append(result)

        # Unmatched AI
        for ai_idx in unmatched_ai:
            if ai_dets[ai_idx].confidence < config.LOW_AI_CONFIDENCE:
                # Low confidence AI -> trigger zoom
                box = (ai_dets[ai_idx].x1, ai_dets[ai_idx].y1,
                       ai_dets[ai_idx].x2, ai_dets[ai_idx].y2)
                zoomed = self._zoom_recheck(frame, box, source="ai_low")
                if zoomed:
                    fused_detections.append(zoomed)
            else:
                result = self._fuse_decision(ai_dets[ai_idx], None, frame)
                if result:
                    fused_detections.append(result)

        # Unmatched CV
        for cv_idx in unmatched_cv:
            result = self._fuse_decision(None, cv_dets[cv_idx], frame)
            if result:
                fused_detections.append(result)

        # Stage 4: Temporal persistence
        tracked_objects = self._update_tracking(fused_detections)
        confirmed, possible = self._classify_hits(tracked_objects)
        self.last_confirmed_hits = confirmed
        self.last_possible_hits = possible

        debug_info["fused_count"] = len(fused_detections)
        debug_info["confirmed_count"] = len(confirmed)
        debug_info["possible_count"] = len(possible)

        # Draw annotations
        annotated = self._draw_annotations(frame, ai_dets, cv_dets, confirmed, possible)

        return annotated, confirmed, debug_info, mask

    def _draw_annotations(self, frame: np.ndarray, ai_dets: List[Detection],
                          cv_dets: List[Detection], confirmed: List[TrackedObject],
                          possible: List[TrackedObject]) -> np.ndarray:
        """Draw all detection boxes with color coding."""
        output = frame.copy()

        # Draw AI detections (thin green)
        for det in ai_dets:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), config.COLOR_AI, 1)
            cv2.putText(output, f"AI:{det.confidence:.2f}", (det.x1, det.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, config.COLOR_AI, 1)

        # Draw CV detections (thin orange)
        for det in cv_dets:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), config.COLOR_CV, 1)
            cv2.putText(output, f"CV:{det.confidence:.2f}", (det.x1, det.y2 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, config.COLOR_CV, 1)

        # Draw fused/confirmed (thick yellow)
        for obj in confirmed:
            det = obj.detection
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), config.COLOR_FUSED, 2)
            cv2.putText(output, f"#{obj.id} {det.confidence:.2f}",
                       (det.x1, det.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, config.COLOR_FUSED, 2)

        # Draw possible hits (thinner purple) for future downstream logic.
        for obj in possible:
            det = obj.detection
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), config.COLOR_POSSIBLE, 1)
            cv2.putText(output, f"P#{obj.id} {obj.fused_confidence:.2f}",
                       (det.x1, det.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, config.COLOR_POSSIBLE, 1)

        # Status text
        cv2.putText(output, f"Frame {self.frame_count} | Hits: {len(confirmed)} | Possible: {len(possible)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output


def run_webcam():
    """Live webcam mode."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set higher resolution for better distant detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fusion = FusionEngine()
    print("Webcam started. Press 'q' to quit, 'd' to toggle debug")

    debug_windows = config.SHOW_DEBUG_WINDOWS
    fps_timer = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, confirmed, debug, mask = fusion.process_frame(frame)

        # FPS calculation
        frame_count += 1
        if time.time() - fps_timer > 1.0:
            fps = frame_count
            print(f"FPS: {fps} | AI: {debug['ai_count']} CV: {debug['cv_count']} "
                  f"Fused: {debug['fused_count']} Hits: {debug['confirmed_count']} "
                  f"Possible: {debug['possible_count']}")
            frame_count = 0
            fps_timer = time.time()

        cv2.imshow("Strawberry Detection", annotated)

        if debug_windows:
            cv2.imshow("CV Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_windows = not debug_windows
            if not debug_windows:
                cv2.destroyWindow("CV Mask")

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path: str = None):
    """Single image mode."""
    import os
    if image_path is None:
        image_path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    fusion = FusionEngine()
    annotated, confirmed, debug, mask = fusion.process_frame(frame)

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
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        img_path = sys.argv[2] if len(sys.argv) > 2 else None
        run_image(img_path)
    else:
        run_webcam()
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from detection import Detection

import config


X_THRESHOLD     = 25
Y_THRESHOLD     = 25
PRIORITIZE_Y    = True

# ── Lock hysteresis ────────────────────────────────────────────────────────────
# A new candidate must beat the current target by this many pixels (baseline).
# The effective threshold is multiplied by the current target's depth_score,
# so a close/confident target is harder to steal than a distant/uncertain one.
LOCK_HYSTERESIS_BASE  = 40    # pixels — baseline steal threshold
LOCK_HYSTERESIS_DEPTH = 30    # extra pixels added at full depth_score (1.0)
#   effective = LOCK_HYSTERESIS_BASE + LOCK_HYSTERESIS_DEPTH * current_depth_score

# ── Ghost-lock grace period ────────────────────────────────────────────────────
# If the locked target is not found in the current frame, hold the lock for up
# to this many consecutive misses before releasing.  Prevents thrashing on
# single-frame detection gaps.
LOCK_GHOST_FRAMES = 2

# ── Candidate scoring weights ──────────────────────────────────────────────────
# Initial target selection (and lock-steal evaluation) ranks candidates by a
# combined score rather than raw pixel distance alone.
#   score = distance_norm * WEIGHT_DIST + (1 - depth_score) * WEIGHT_DEPTH
# Both terms are "lower is better", so the candidate with the lowest score wins.
WEIGHT_DIST  = 0.5   # how much pixel distance matters
WEIGHT_DEPTH = 0.5   # how much depth (proximity) matters

# ── Depth units ───────────────────────────────────────────────────────────────
# depth_units is a relative distance value where 1.0 means the berry appears at
# exactly the ideal (picking) size.  Values > 1.0 mean farther away; < 1.0 mean
# closer than ideal.
#
# Formula:  depth_units = sqrt(BERRY_SIZE_IDEAL / apparent_area)
#
#
# Clipped to [DEPTH_UNITS_MIN, DEPTH_UNITS_MAX] so extreme noise doesn't
# produce absurd values.
DEPTH_UNITS_MIN = 0.2
DEPTH_UNITS_MAX = 1000

# =========================
# DEPTH ESTIMATION CONFIG
# =========================
DEPTH_CONF_WEIGHT = 0.35   # how much detection confidence softens the score (0 = ignore)


@dataclass
class RobotTarget:
    detection:   Detection
    center_x:    int
    center_y:    int
    distance:    float
    depth_score: float = 0.0   # 0–1, 1 = at ideal picking distance
    depth_units: float = 1.0   # relative distance, 1.0 = ideal picking distance


class RobotController:
    """Select target strawberries and produce simple movement commands."""

    def __init__(self) -> None:
        self.current_target: Optional[RobotTarget] = None
        self._ghost_frames: int = 0   # consecutive frames the target was not found

    # ── geometry helpers ───────────────────────────────────────────────────────

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        return (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2

    @staticmethod
    def _distance_to(gripper_x: int, gripper_y: int, x: int, y: int) -> float:
        return math.hypot(x - gripper_x, y - gripper_y)

    # ── depth estimation ───────────────────────────────────────────────────────

    @staticmethod
    def estimate_depth(det: Detection) -> float:
        """
        Estimate relative depth from apparent bounding-box size.

        Returns depth_score in [0.0, 1.0]:
            1.0  →  berry is close (at or above ideal size)
            0.0  →  berry is far away (much smaller than ideal)

        Asymmetric curve:
            - Smaller than ideal → score drops toward 0 (far)
            - Larger than ideal  → score stays near 1.0 (very close)
        """
        area  = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)
        ratio = area / ideal  # <1 = far, >1 = close

        if ratio >= 1.0:
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 1.5 ** 2))
        else:
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 0.35 ** 2))

        conf  = float(det.confidence or 0.0)
        score = raw_score * (1.0 - DEPTH_CONF_WEIGHT) + raw_score * conf * DEPTH_CONF_WEIGHT
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def estimate_depth_units(det: Detection) -> float:
        """
        Return a relative distance value where 1.0 = berry at ideal picking size.

        depth_units = sqrt(BERRY_SIZE_IDEAL / apparent_area)

        Properties
        ----------
        - Physically linear with real distance (area ∝ 1/d², so sqrt inverts it).
        - 20→60 unit real-world range gives a 3× swing
        - Clipped to [DEPTH_UNITS_MIN, DEPTH_UNITS_MAX] to suppress noise.
        - Values are always positive; lower = closer.
        """
        area  = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)
        units = math.sqrt(ideal / area)
        return round(min(DEPTH_UNITS_MAX, max(DEPTH_UNITS_MIN, units)), 3)

    @staticmethod
    def depth_label(depth_score: float) -> str:
        """Human-readable depth bucket for logging."""
        if depth_score >= 0.80:
            return "CLOSE"
        if depth_score >= 0.50:
            return "MEDIUM"
        if depth_score >= 0.25:
            return "FAR"
        return "VERY FAR"

    # ── candidate scoring ──────────────────────────────────────────────────────

    def _candidate_score(
        self,
        dist: float,
        depth_score: float,
        max_dist: float,
    ) -> float:
        """
        Combined score for candidate ranking — lower is better.

        Normalises pixel distance to [0, 1] using the furthest candidate in
        this frame as the reference, then blends with (1 - depth_score).
        """
        dist_norm = dist / max(max_dist, 1.0)
        return dist_norm * WEIGHT_DIST + (1.0 - depth_score) * WEIGHT_DEPTH

    # ── lock management ────────────────────────────────────────────────────────

    def _target_still_exists(self, detections: List[Detection]) -> bool:
        """
        Try to match the current target against new detections.

        Returns True (and updates the target in-place) if found.
        If not found, increments the ghost-frame counter and returns True
        while still within the grace period, so the lock is held temporarily.
        """
        if self.current_target is None:
            return False

        target = self.current_target.detection
        for det in detections:
            if self._simple_iou(target, det) > 0.3:
                cx, cy = self.get_box_center(det)
                self.current_target.detection   = det
                self.current_target.center_x    = cx
                self.current_target.center_y    = cy
                self.current_target.distance    = 0.0
                self.current_target.depth_score = self.estimate_depth(det)
                self.current_target.depth_units = self.estimate_depth_units(det)
                self._ghost_frames = 0
                return True

        # Not found — apply ghost-lock grace period
        self._ghost_frames += 1
        if self._ghost_frames <= LOCK_GHOST_FRAMES:
            return True   # hold the lock without updating position

        # Grace period expired — release
        self._ghost_frames = 0
        return False

    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:
        x_a   = max(a.x1, b.x1)
        y_a   = max(a.y1, b.y1)
        x_b   = min(a.x2, b.x2)
        y_b   = min(a.y2, b.y2)
        inter  = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ── public API ─────────────────────────────────────────────────────────────

    def choose_target(
        self,
        detections: List[Detection],
        gripper_x: int,
        gripper_y: int,
    ) -> Optional[RobotTarget]:

        if not detections:
            self.current_target = None
            self._ghost_frames  = 0
            return None

        # Build candidates with full metrics
        candidates: List[RobotTarget] = []
        for det in detections:
            cx, cy = self.get_box_center(det)
            dist   = self._distance_to(gripper_x, gripper_y, cx, cy)
            depth  = self.estimate_depth(det)
            units  = self.estimate_depth_units(det)
            candidates.append(RobotTarget(det, cx, cy, dist, depth, units))

        # max_dist reference includes all candidates so normalisation is stable
        max_dist = max(c.distance for c in candidates)

        # Find the best candidate by combined score
        best = min(
            candidates,
            key=lambda c: self._candidate_score(c.distance, c.depth_score, max_dist),
        )

        # Update current target's metrics (or apply ghost-lock) before comparing
        target_alive = self._target_still_exists(detections)

        if target_alive:
            # Re-score the (now freshly updated) current target on the same
            # max_dist scale so the comparison is apples-to-apples.
            cur = self.current_target
            cur_dist = self._distance_to(gripper_x, gripper_y, cur.center_x, cur.center_y)
            cur_score  = self._candidate_score(cur_dist, cur.depth_score, max_dist)
            best_score = self._candidate_score(best.distance, best.depth_score, max_dist)

            # Hysteresis: expressed as a score-space margin so depth differences
            # are properly accounted for.  Deeper/closer lock → larger margin.
            # LOCK_HYSTERESIS_BASE / max_dist normalises pixels → score space.
            margin = (
                (LOCK_HYSTERESIS_BASE + LOCK_HYSTERESIS_DEPTH * cur.depth_score)
                / max(max_dist, 1.0)
            ) * WEIGHT_DIST   # only the distance component is in comparable units

            if best_score < cur_score - margin:
                self._ghost_frames  = 0
                self.current_target = best

            return self.current_target

        # No existing target — take the best combined-score candidate
        self._ghost_frames  = 0
        self.current_target = best
        return self.current_target

    def generate_dx(self, gripper_x: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_x - gripper_x

    def generate_dy(self, gripper_y: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_y - gripper_y

    def generate_movementstring(self, gripper_x: int, gripper_y: int) -> str:
        if self.current_target is None:
            return "NO TARGET"

        tx = self.current_target.center_x
        ty = self.current_target.center_y

        dx = tx - gripper_x
        dy = ty - gripper_y

        if PRIORITIZE_Y:
            if abs(dy) > Y_THRESHOLD:
                return "ARM GO DOWN" if dy > 0 else "ARM GO UP"
        if abs(dx) > X_THRESHOLD:
            return "MOVE RIGHT" if dx > 0 else "MOVE LEFT"

        return "TARGET LOCKED"

    def generate_depthstring(self) -> str:
        """One-liner depth summary for the current target, ready to log."""
        if self.current_target is None:
            return "DEPTH: no target"
        t    = self.current_target
        area = (t.detection.x2 - t.detection.x1) * (t.detection.y2 - t.detection.y1)
        return (
            f"DEPTH: {self.depth_label(t.depth_score)} "
            f"(dist={t.depth_units:.2f}, "
            f"area={area}px²)"
        )
import os
from typing import Tuple

import cv2
import numpy as np

import config
from detection import AIDetector, CVDectector


_AI = AIDetector()
_CV = CVDectector()


def _draw_boxes(frame: np.ndarray, detections, color: Tuple[int, int, int], prefix: str) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        cv2.putText(out, f"{prefix}:{det.confidence:.2f}", (det.x1, max(14, det.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def cv_pipeline(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (annotated_frame, mask, distance_view, berry_count)."""
    detections, mask = _CV.detect(frame)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_view = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return _draw_boxes(frame, detections, config.COLOR_CV, "CV"), mask, dist_view, len(detections)


def ai_pipeline(frame: np.ndarray, conf: float = 0.5) -> tuple[np.ndarray, int]:
    detections = _AI.detect(frame, conf_threshold=conf)
    return _draw_boxes(frame, detections, config.COLOR_AI, "AI"), len(detections)


def run_webcam(frames: bool) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found.")
        return

    print("Webcam started (press 'q' to quit).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv_out, mask, dist_view, cv_count = cv_pipeline(frame)
        ai_out, ai_count = ai_pipeline(frame)

        combined = frame.copy()
        cv2.putText(combined, f"CV: {cv_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_CV, 2)
        cv2.putText(combined, f"AI: {ai_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_AI, 2)

        if frames:
            cv2.imshow("Combined", combined)
            cv2.imshow("AI", ai_out)
            cv2.imshow("CV", cv_out)
            cv2.imshow("Mask", mask)
            cv2.imshow("Distance", dist_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image() -> None:
    img_path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    cv_out, _, dist_view, count = cv_pipeline(frame)
    cv2.imshow("Distance", dist_view)
    cv2.imshow("Result", cv_out)
    print(f"Detected strawberries: {count}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam(True)
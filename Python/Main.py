"""Entry points for the strawberry fusion detector."""

import os
import time
from typing import Optional

import cv2
import numpy as np

import config
from fusion_engine import DETECT_EVERY, INFER_SCALE, FusionEngine


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def _connect_camera(rtsp_url: str) -> cv2.VideoCapture:
    print(f"Connecting to reCamera stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                print("reCamera stream connected.")
                return cap
            time.sleep(0.1)

    print("reCamera not available - falling back to laptop camera.")
    cap.release()
    return _open_laptop_camera()


def _open_laptop_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("No camera available.")
    print("Laptop camera connected.")
    return cap


def choose_camera_mode() -> bool:
    """Returns True = use reCamera, False = use laptop camera."""
    while True:
        choice = input("Camera: [r] reCamera  [l] Laptop  → ").strip().lower()
        if choice == "r":
            return True
        if choice == "l":
            return False
        print("  Please type 'r' or 'l'.")


def run_webcam(rtsp_url: str = "rtsp://admin:admin@192.168.42.1:554/live") -> None:
    fusion = FusionEngine()



    show_mask = config.SHOW_DEBUG_WINDOWS
    fps_timer = time.perf_counter()
    fps_count = 0
    cap = None

    try:
        if choose_camera_mode():
            cap = _connect_camera(rtsp_url)
        else:
            cap = _open_laptop_camera()

        print(f"\nInference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")
        print("Press 'q' to quit, 'd' to toggle debug mask.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame, retrying...")
                continue

            annotated, _, debug, mask = fusion.process_frame(frame)

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

            cv2.imshow("Strawberry Detection",
                       cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))

            if show_mask and mask is not None:
                cv2.imshow("CV Mask",
                           cv2.resize(mask, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("CV Mask")

    finally:
        fusion.shutdown()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def run_image(image_path: Optional[str] = None) -> None:
    if image_path is None:
        image_path = os.path.join(os.path.dirname(__file__), "..", "Assets", "StrawberryPlant1Full.jpg")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    fusion = FusionEngine()
    fusion.process_frame(np.asarray(frame))
    time.sleep(0.5)
    annotated, confirmed, debug, mask = fusion.process_frame(np.asarray(frame))
    possible = fusion.last_possible_hits
    fusion.shutdown()

    print(f"\nResults for {image_path}:")
    print(
        f"  AI: {debug['ai_count']} | CV: {debug['cv_count']} | "
        f"Fused: {debug['fused_count']} | Hits: {debug['confirmed_count']} "
        f"| Possible: {debug['possible_count']}"
    )

    for obj in confirmed:
        det = obj.detection
        print(f"  Berry {obj.id}: conf={det.confidence:.3f}, seen={obj.seen_count} frames, source={det.source}")

    if possible:
        print("  Possible hits:")
        for obj in possible:
            det = obj.detection
            print(f"    P{obj.id}: conf={obj.fused_confidence:.3f}, seen={obj.seen_count} frames, source={det.source}")

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
        run_webcam(sys.argv[1] if len(sys.argv) > 1 else "rtsp://admin:admin@192.168.42.1:554/live")
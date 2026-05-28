"""Entry points for the strawberry fusion detector."""

import os
import time
from typing import Optional

import cv2
import numpy as np

import config
import web_server
from fusion_engine import DETECT_EVERY, INFER_SCALE, FusionEngine

import os, sys
print(f"isatty={sys.stdin.isatty()}, stdin fd={sys.stdin.fileno()}, pid={os.getpid()}", flush=True)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024000"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

RTSP_ETHERNET = "rtsp://admin:admin@169.254.192.21:554/live"
RTSP_USB      = "rtsp://admin:admin@192.168.42.1:554/live"


# ── camera helpers ─────────────────────────────────────────────────────────────

def _try_rtsp(rtsp_url: str, label: str) -> Optional[cv2.VideoCapture]:
    """Probeer een RTSP stream te openen. Geeft None terug als het mislukt."""
    print(f"Probeer {label}: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"{label} verbonden!")
                return cap
            time.sleep(0.1)

    print(f"{label} niet beschikbaar.")
    cap.release()
    return None


def _connect_camera() -> tuple[cv2.VideoCapture, str]:
    cap = _try_rtsp(RTSP_ETHERNET, "reCamera Ethernet")
    if cap is not None:
        return cap, "Ethernet"

    cap = _try_rtsp(RTSP_USB, "reCamera USB")
    if cap is not None:
        return cap, "USB"

    return _open_laptop_camera(), "Laptop"


def _open_laptop_camera() -> cv2.VideoCapture:
    print("Geen reCamera gevonden — laptop camera proberen...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Geen enkele camera beschikbaar.")
    print("Laptop camera verbonden.")
    return cap


# ── camera badge ───────────────────────────────────────────────────────────────

def _draw_camera_badge(frame: np.ndarray, mode: str) -> None:
    """Draw a small camera-mode label in the top-right corner (in-place)."""
    label   = f"CAM: {mode}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1
    padding = 6

    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    h, w = frame.shape[:2]

    x1 = w - tw - padding * 2 - 2
    y1 = 2
    x2 = w - 2
    y2 = th + baseline + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    colours = {"Ethernet": (80, 220, 80), "USB": (60, 220, 220), "Laptop": (220, 220, 60)}
    dot_colour = colours.get(mode, (200, 200, 200))
    dot_x = x1 + padding + 5
    dot_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (dot_x, dot_y), 4, dot_colour, cv2.FILLED)

    tx = dot_x + 10
    ty = y1 + padding + th
    cv2.putText(frame, label, (tx, ty), font, scale, (230, 230, 230), thick, cv2.LINE_AA)


# ── run_webcam ─────────────────────────────────────────────────────────────────

def run_webcam() -> None:
    web_server.start()

    fusion = FusionEngine()

    show_mask = config.SHOW_DEBUG_WINDOWS
    fps_timer = time.perf_counter()
    fps_count = 0
    cap = None

    try:
        cap, cam_mode = _connect_camera()

        print(f"\nCamera mode: {cam_mode}")
        print(f"Inference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")
        print("Press 'q' to quit, 'd' to toggle debug mask.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame, retrying...")
                continue

            annotated, _, debug, mask = fusion.process_frame(frame)

            display = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            _draw_camera_badge(display, cam_mode)

            web_server.push_frame(display)

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

            cv2.imshow("Strawberry Detection", display)

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
        run_webcam()
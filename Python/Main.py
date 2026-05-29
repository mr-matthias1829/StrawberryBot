"""Entry points for the strawberry fusion detector."""

import os
# Set OpenCV FFmpeg capture options early so they are available to cv2 when it
# initializes its backends. Keep this lightweight and non-fatal if not used.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp|buffer_size;1024000")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import time

import platform
ON_PI = platform.system() == "Linux"
if ON_PI:
    import motor
    motor.init()

import sys
try:
    isatty = sys.stdin.isatty()
    try:
        stdin_fd = sys.stdin.fileno()
    except Exception:
        stdin_fd = None
except Exception:
    # In some embedded/headless runtimes stdin may be unavailable.
    isatty = False
    stdin_fd = None

print(f"isatty={isatty}, stdin fd={stdin_fd}, pid={os.getpid()}", flush=True)

import cv2
import numpy as np

import config
import web_server
from fusion_engine import DETECT_EVERY, INFER_SCALE, FusionEngine

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

RTSP_ETHERNET = "rtsp://admin:admin@169.254.192.21:554/live"
RTSP_USB      = "rtsp://admin:admin@192.168.42.1:554/live"


@dataclass
class _WorkerState:
    frame: Optional[np.ndarray] = None
    result: Optional[tuple[np.ndarray, list, dict, Optional[np.ndarray]]] = None
    lock: object = None
    stop: bool = False


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

    # Compute a small rectangle region in the top-right corner and draw only
    # into that ROI to avoid copying the whole frame each frame.
    x2 = w - 2
    x1 = max(0, x2 - (tw + padding * 2 + 10))
    y1 = 2
    y2 = min(h, y1 + th + baseline + padding * 2)

    # Darken the ROI in-place (approximate the overlay effect) by directly
    # blending the region without copying the full image.
    roi = frame[y1:y2, x1:x2]
    if roi.size != 0:
        dark = (20, 20, 20)
        alpha = 0.55
        # Multiply existing region by (1-alpha) and add dark*alpha
        roi[:] = (roi.astype('float32') * (1 - alpha) + np.array(dark, dtype='float32') * alpha).astype('uint8')

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
    import threading

    show_mask = config.SHOW_DEBUG_WINDOWS
    fps_timer = time.perf_counter()
    fps_count = 0
    cap = None
    headless = False
    read_fail_count = 0
    READ_FAIL_MAX = 30

    # Shared state between capture loop and inference worker.
    state = _WorkerState(lock=threading.Lock())

    def inference_worker(st: _WorkerState):
        """Worker thread that owns a FusionEngine instance and processes frames
        from the shared state. It keeps only the most recent frame, so slow
        inference won't pile up a queue.
        """
        try:
            engine = FusionEngine()
        except Exception as e:
            print(f"Failed to initialize FusionEngine in worker: {e}")
            return

        try:
            while not st.stop:
                frame = None
                with cast(threading.Lock, st.lock):
                    if st.frame is not None:
                        frame = st.frame
                        st.frame = None
                if frame is None:
                    time.sleep(0.005)
                    continue

                frame = cast(np.ndarray, frame)
                try:
                    annotated, confirmed, debug, mask = engine.process_frame(frame)
                except Exception as e:
                    print(f"Inference worker: process_frame error: {e}")
                    continue

                with cast(threading.Lock, st.lock):
                    st.result = (annotated, confirmed, debug, mask)
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass

    worker = threading.Thread(target=inference_worker, args=(state,), daemon=True, name="inference-worker")
    worker.start()

    try:
        cap, cam_mode = _connect_camera()

        print(f"\nCamera mode: {cam_mode}")
        print(f"Inference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")
        print("Press 'q' to quit, 'd' to toggle debug mask.\n")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_fail_count += 1
                print("Dropped frame, retrying... (failure %d)" % read_fail_count)
                # small backoff to avoid busy-looping
                time.sleep(0.05)
                if read_fail_count >= READ_FAIL_MAX:
                    print("Too many consecutive read failures, attempting reconnect...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        cap, cam_mode = _connect_camera()
                        read_fail_count = 0
                        print(f"Reconnected, camera mode: {cam_mode}")
                    except Exception as e:
                        print(f"Reconnect failed: {e}")
                        time.sleep(1.0)
                continue

            # reset on success
            read_fail_count = 0

            # Feed the latest frame to the worker (non-blocking, overwrites previous)
            with cast(threading.Lock, state.lock):
                state.frame = frame

            # Use latest processed result if available; otherwise continue showing
            # the previous annotated frame to avoid black frames while inference
            # is busy. We copy the reference under lock to a local variable.
            res = None
            with cast(threading.Lock, state.lock):
                res = state.result

            if res is None:
                # No processed frame yet; skip updating display (keeps last image)
                continue

            annotated, _, debug, mask = res
            if annotated is None:
                continue

            # Only resize if needed (saves CPU when sizes already match)
            try:
                h, w = annotated.shape[:2]
            except Exception:
                continue

            if (w, h) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
                display = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            else:
                display = annotated

            display = np.asarray(display)

            _draw_camera_badge(display, cam_mode)

            try:
                web_server.push_frame(display)
            except Exception as e:
                # Don't let web server push failures kill the main loop
                print(f"Warning: web_server.push_frame failed: {e}")

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

            # GUI operations can fail on headless systems; guard them and
            # switch to headless mode if imshow/waitKey raise errors.
            if not headless:
                try:
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
                except Exception as e:
                    print(f"GUI unavailable (switching to headless): {e}")
                    headless = True

    finally:
        # Stop worker and join
        state.stop = True
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def run_image(image_path: Optional[str] = None) -> None:
    if image_path is None:
        image_path = str((Path(__file__).resolve().parent / ".." / "Assets" / "StrawberryPlant1Full.jpg").resolve())

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    fusion = FusionEngine()
    annotated = None
    confirmed = []
    debug = {"ai_count": 0, "cv_count": 0, "fused_count": 0, "confirmed_count": 0, "possible_count": 0}
    mask = None
    possible = None

    try:
        # warmup (non-fatal if it fails)
        try:
            fusion.process_frame(np.asarray(frame))
        except Exception:
            pass
        time.sleep(0.5)
        annotated, confirmed, debug, mask = fusion.process_frame(np.asarray(frame))
        possible = fusion.last_possible_hits
    except Exception as e:
        print(f"Error during image processing: {e}")
        fusion.shutdown()
        return
    finally:
        # ensure resources are freed even on error
        try:
            fusion.shutdown()
        except Exception:
            pass

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

    try:
        cv2.imshow("Result", annotated)
        if mask is not None:
            cv2.imshow("Mask", mask)
        cv2.waitKey(0)
    except Exception as e:
        print(f"GUI unavailable for image display: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        run_image(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        run_webcam()



def _motor_startup_test():
    """Quick sanity check — moves ID 13 left/right, then gripper close/open."""
    print("\n── Motor startup test ──")
    try:
        from gripper import Gripper

        motor.enable_torque(13)

        result = motor.turn_left(servo_id=13)
        print(f"  ID 13 left:  {result['status']}")

        result = motor.turn_right(servo_id=13)
        print(f"  ID 13 right: {result['status']}")

        motor.disable_torque(13)

        g = Gripper()
        g.grip()
        g.open()

    except Exception as e:
        print(f"  ⚠️  Motor test failed (continuing anyway): {e}")

    print("── Motor test done ──\n")

_motor_startup_test()
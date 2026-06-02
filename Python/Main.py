"""Entry points for the strawberry fusion detector."""

import os
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|fflags;discardcorrupt|analyzeduration;500000|probesize;500000"
)
import time
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import cv2
import numpy as np

ON_PI = platform.system() == "Linux"

# =============================================================================
# Hardware init — must happen before any submodule that touches motor/servos
# =============================================================================
if ON_PI:
    import motor
    motor.init()

    import turntable
    turntable.init()

    import lift
    lift.init()

    import gripper
    gripper.init()

# =============================================================================
# Stdin diagnostics (harmless on all platforms)
# =============================================================================
try:
    isatty   = sys.stdin.isatty()
    stdin_fd = sys.stdin.fileno() if hasattr(sys.stdin, "fileno") else None
except Exception:
    isatty   = False
    stdin_fd = None

print(f"isatty={isatty}, stdin fd={stdin_fd}, pid={os.getpid()}", flush=True)

import config
import web_server
from fusion_engine import DETECT_EVERY, INFER_SCALE, FusionEngine

DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

RTSP_ETHERNET = "rtsp://admin:admin@169.254.192.21:554/live"
RTSP_USB      = "rtsp://admin:admin@192.168.42.1:554/live"


# =============================================================================
# Real-Time Robust Capture Thread (Auto-reconnects on drop)
# =============================================================================

class Capture:
    def __init__(self, rtsp_url: str, label: str):
        self.rtsp_url = rtsp_url
        self.label = label
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.lock = threading.Lock()

        # Initial connection attempt
        self._establish_connection()

        self.thread = threading.Thread(target=self._reader, daemon=True, name="camera-buffer-flusher")
        self.thread.start()

    def _establish_connection(self) -> None:
        """Safely opens or re-opens the cv2 VideoCapture pipeline."""
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            print(f"[{self.label}] Connecting to stream: {self.rtsp_url}...")
            if self.rtsp_url == "0":
                # Handle local laptop camera fallback
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            else:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.cap = cap

    def _reader(self) -> None:
        consecutive_failures = 0
        MAX_FAILURES = 15 # ~3-4 seconds of dead frames before triggering hard restart

        while self.running:
            # Check if capture object exists
            with self.lock:
                current_cap = self.cap

            if current_cap is None or not current_cap.isOpened():
                print(f"[{self.label}] Capture object is dead or null. Reconnecting...")
                self._establish_connection()
                time.sleep(1.0)
                continue

            # Attempt to grab the frame from hardware buffer
            if not current_cap.grab():
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[{self.label}] Stream died (failed grab). Hard reconnecting pipeline...")
                    self._establish_connection()
                    consecutive_failures = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.01)
                continue

            # Retrieve the grab
            ok, frame = current_cap.retrieve()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[{self.label}] Stream returned null frame data. Hard reconnecting...")
                    self._establish_connection()
                    consecutive_failures = 0
                    time.sleep(1.0)
                continue

            # Successful capture loop cycle
            consecutive_failures = 0
            with self.lock:
                self.frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame

    def release(self) -> None:
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        with self.lock:
            if self.cap is not None:
                self.cap.release()


# =============================================================================
# Worker state
# =============================================================================

@dataclass
class _WorkerState:
    frame:  Optional[np.ndarray] = None
    result: Optional[tuple]      = None
    lock:   object               = None
    stop:   bool                 = False


# =============================================================================
# Camera finder helper
# =============================================================================

def _find_available_camera_source() -> tuple[str, str]:
    """Probes URLs to find out which camera source is online."""
    for url, label in [(RTSP_ETHERNET, "reCamera Ethernet"), (RTSP_USB, "reCamera USB")]:
        print(f"Probing {label}...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                print(f"Found active stream on {label}")
                return url, label
    print("No reCamera discovered. Falling back to laptop camera environment.")
    return "0", "Laptop"


# =============================================================================
# Camera badge
# =============================================================================

def _draw_camera_badge(frame: np.ndarray, mode: str) -> None:
    label   = f"CAM: {mode}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1
    padding = 6
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    h, w = frame.shape[:2]

    x2 = w - 2
    x1 = max(0, x2 - (tw + padding * 2 + 10))
    y1 = 2
    y2 = min(h, y1 + th + baseline + padding * 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size != 0:
        roi[:] = (
            roi.astype("float32") * 0.45
            + np.array([20, 20, 20], dtype="float32") * 0.55
        ).astype("uint8")

    colours    = {"reCamera Ethernet": (80, 220, 80), "reCamera USB": (60, 220, 220), "Laptop": (220, 220, 60)}
    dot_colour = colours.get(mode, (200, 200, 200))
    dot_x = x1 + padding + 5
    dot_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (dot_x, dot_y), 4, dot_colour, cv2.FILLED)
    cv2.putText(frame, label, (dot_x + 10, y1 + padding + th),
                font, scale, (230, 230, 230), thick, cv2.LINE_AA)


# =============================================================================
# Inference worker thread
# =============================================================================

def _inference_worker(st: _WorkerState) -> None:
    """Owns a FusionEngine; processes the latest frame posted by the capture loop."""
    try:
        engine = FusionEngine()
    except Exception as e:
        print(f"Failed to init FusionEngine in worker: {e}")
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

            try:
                t0 = time.perf_counter()
                annotated, confirmed, debug, mask = engine.process_frame(frame)
                process_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                print(f"[inference] process_frame error: {e}")
                continue

            with cast(threading.Lock, st.lock):
                st.result = (annotated, confirmed, debug, mask, process_ms)

    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


# =============================================================================
# run_webcam
# =============================================================================

def run_webcam() -> None:
    web_server.start()

    show_mask       = config.SHOW_DEBUG_WINDOWS
    fps_timer       = time.perf_counter()
    fps_count       = 0
    capture_wrapper = None
    headless        = False

    state = _WorkerState(lock=threading.Lock())
    worker = threading.Thread(
        target=_inference_worker,
        args=(state,),
        daemon=True,
        name="inference-worker",
    )
    worker.start()

    try:
        # Probe network strings and initialize auto-healing Capture wrapper
        target_url, cam_mode = _find_available_camera_source()
        capture_wrapper = Capture(target_url, cam_mode)

        print(f"\nCamera Engine initialized in background thread context.")
        print(f"Inference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")
        print("Press 'q' to quit, 'd' to toggle debug mask.\n")

        last_read_ms = 0.0
        last_proc_ms = 0.0

        while True:
            t_read = time.perf_counter()
            frame = capture_wrapper.read()
            last_read_ms = (time.perf_counter() - t_read) * 1000.0

            if frame is None:
                # Thread is actively reconnecting or waiting for first frame buffer to arrive
                time.sleep(0.03)
                continue

            # Pass the frame safely down to the worker thread
            with cast(threading.Lock, state.lock):
                state.frame = frame

            # Safely request last built result matrix
            res = None
            with cast(threading.Lock, state.lock):
                res = state.result

            if res is None:
                # Keep spinning camera collection loops until inference spins up
                continue

            annotated, _, debug, mask, last_proc_ms = res
            if annotated is None:
                continue

            # Resize rendering layout
            h, w = annotated.shape[:2]
            display = (
                cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                if (w, h) != (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                else annotated
            )
            display = np.asarray(display)
            _draw_camera_badge(display, cam_mode)

            try:
                web_server.push_frame(display)
            except Exception as e:
                print(f"Warning: web_server.push_frame failed: {e}")

            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                print(
                    f"FPS: {fps_count:2d} | "
                    f"Read: {last_read_ms:6.1f} ms | "
                    f"Proc: {last_proc_ms:6.1f} ms | "
                    f"AI: {debug['ai_count']} CV: {debug['cv_count']} "
                    f"Fused: {debug['fused_count']} "
                    f"Hits: {debug['confirmed_count']} "
                    f"Possible: {debug['possible_count']}"
                )
                fps_count = 0
                fps_timer = now

            if not headless:
                try:
                    cv2.imshow("Strawberry Detection", display)
                    if show_mask and mask is not None:
                        cv2.imshow(
                            "CV Mask",
                            cv2.resize(mask, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)),
                        )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d"):
                        show_mask = not show_mask
                        if not show_mask:
                            cv2.destroyWindow("CV Mask")
                except Exception as e:
                    print(f"GUI unavailable (headless): {e}")
                    headless = True

    finally:
        state.stop = True
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
        if capture_wrapper is not None:
            capture_wrapper.release()
        cv2.destroyAllWindows()
        if ON_PI:
            gripper.shutdown()
            lift.shutdown()
            turntable.shutdown()
            motor.shutdown() # tries to shutdown motors activated through its class
                            # consider removing motor shutdown as we dont activate motors through motor.py

# =============================================================================
# run_image
# =============================================================================

def run_image(image_path: Optional[str] = None) -> None:
    if image_path is None:
        image_path = str(
            (Path(__file__).resolve().parent / ".." / "Assets" / "StrawberryPlant1Full.jpg").resolve()
        )

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    fusion    = FusionEngine()
    annotated = None
    confirmed = []
    debug     = {"ai_count": 0, "cv_count": 0, "fused_count": 0, "confirmed_count": 0, "possible_count": 0}
    mask      = None
    possible  = None

    try:
        try:
            fusion.process_frame(np.asarray(frame))   # warmup
        except Exception:
            pass
        time.sleep(0.5)
        annotated, confirmed, debug, mask = fusion.process_frame(np.asarray(frame))
        possible = fusion.last_possible_hits
    except Exception as e:
        print(f"Error during image processing: {e}")
        return
    finally:
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
        print(f"  Berry {obj.id}: conf={det.confidence:.3f}, seen={obj.seen_count}, source={det.source}")
    if possible:
        print("  Possible hits:")
        for obj in possible:
            det = obj.detection
            print(f"    P{obj.id}: conf={obj.fused_confidence:.3f}, seen={obj.seen_count}, source={det.source}")

    try:
        cv2.imshow("Result", annotated)
        if mask is not None:
            cv2.imshow("Mask", mask)
        cv2.waitKey(0)
    except Exception as e:
        print(f"GUI unavailable: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--image":
        run_image(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        run_webcam()
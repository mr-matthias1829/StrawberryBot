"""Entry points for the strawberry fusion detector."""
import os
import time
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import cv2
import numpy as np
import signal

ON_PI = platform.system() == "Linux"

# =============================================================================
# SIGTERM handler
# =============================================================================

def _sigterm_handler(signum, frame):
    print("[main] SIGTERM received — shutting down cleanly…", flush=True)
    if ON_PI:
        for mod in (manual_controller, gripper, arm, pivot, lift, turntable, motor):
            try:
                (mod.stop if mod is manual_controller else mod.shutdown)()
            except Exception:
                pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)

# =============================================================================
# Hardware init
# =============================================================================

if ON_PI:
    import motor;            motor.init()
    import turntable;        turntable.init()
    import lift;             lift.init()
    import arm;              arm.init()
    import pivot;            pivot.init()
    import gripper;          gripper.init()
    import manual_controller; manual_controller.start()

# =============================================================================
# Stdin diagnostics
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
import control_mode
from fusion_engine import DETECT_EVERY, INFER_SCALE, FusionEngine

DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720

# reCamera Seed 200X — UVC device path on the Pi.
# Run `v4l2-ctl --list-devices` to confirm; usually /dev/video0.
UVC_DEVICE = "/dev/video0"


# =============================================================================
# Real-Time Robust Capture Thread (auto-reconnects on drop)
# =============================================================================

class Capture:
    def __init__(self, device: str, label: str):
        self.device = device
        self.label  = label
        self.cap:   Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray]       = None
        self.running = True
        self.lock    = threading.Lock()

        self._establish_connection()
        self.thread = threading.Thread(
            target=self._reader, daemon=True, name="camera-buffer-flusher"
        )
        self.thread.start()

    def _establish_connection(self) -> None:
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            print(f"[{self.label}] Opening UVC device: {self.device}…")

            if ON_PI:
                cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            else:
                # Laptop fallback: index 0, default backend
                cap = cv2.VideoCapture(0)

            cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

            self.cap = cap

    def _reader(self) -> None:
        consecutive_failures = 0
        MAX_FAILURES = 15

        while self.running:
            with self.lock:
                current_cap = self.cap

            if current_cap is None or not current_cap.isOpened():
                print(f"[{self.label}] Capture dead. Reconnecting…")
                self._establish_connection()
                time.sleep(1.0)
                continue

            # V4L2 grab() blocks until the next frame arrives — no busy-wait needed.
            if not current_cap.grab():
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[{self.label}] Grab failed {MAX_FAILURES}×. Reconnecting…")
                    self._establish_connection()
                    consecutive_failures = 0
                    time.sleep(1.0)
                else:
                    time.sleep(0.03)
                continue

            ok, frame = current_cap.retrieve()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[{self.label}] Null frame {MAX_FAILURES}×. Reconnecting…")
                    self._establish_connection()
                    consecutive_failures = 0
                    time.sleep(1.0)
                continue

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
# Camera badge
# =============================================================================

def _draw_camera_badge(frame: np.ndarray, label: str) -> None:
    text    = f"CAM: {label}"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1
    padding = 6
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    h, w = frame.shape[:2]

    x2 = w - 2
    x1 = max(0, x2 - tw - padding * 2 - 10)
    y1 = 2
    y2 = min(h, y1 + th + baseline + padding * 2)

    roi = frame[y1:y2, x1:x2]
    if roi.size:
        roi[:] = (roi.astype("float32") * 0.45 + np.array([20, 20, 20], dtype="float32") * 0.55).astype("uint8")

    dot_x = x1 + padding + 5
    dot_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (dot_x, dot_y), 4, (60, 220, 220), cv2.FILLED)
    cv2.putText(frame, text, (dot_x + 10, y1 + padding + th), font, scale, (230, 230, 230), thick, cv2.LINE_AA)


# =============================================================================
# Control-mode badge
# =============================================================================

def _draw_mode_badge(frame: np.ndarray) -> None:
    is_manual = control_mode.is_manual()
    text      = "MANUAL" if is_manual else "AUTO"
    colour    = (0, 100, 255) if is_manual else (0, 220, 100)

    font    = cv2.FONT_HERSHEY_SIMPLEX
    scale   = 0.55
    thick   = 1
    padding = 6
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

    x1 = 2
    y1 = 2
    x2 = x1 + tw + padding * 2 + 14
    y2 = y1 + th + baseline + padding * 2

    roi = frame[y1:y2, x1:x2]
    if roi.size:
        roi[:] = (roi.astype("float32") * 0.45 + np.array([20, 20, 20], dtype="float32") * 0.55).astype("uint8")

    dot_x = x1 + padding + 5
    dot_y = y1 + (y2 - y1) // 2
    cv2.circle(frame, (dot_x, dot_y), 4, colour, cv2.FILLED)
    cv2.putText(frame, text, (dot_x + 10, y1 + padding + th), font, scale, (230, 230, 230), thick, cv2.LINE_AA)


# =============================================================================
# Inference worker thread
# =============================================================================

def _inference_worker(st: _WorkerState) -> None:
    """Owns a FusionEngine; processes the latest frame posted by the capture loop."""
    try:
        engine = FusionEngine()
    except Exception as e:
        print(f"[inference] Failed to init FusionEngine: {e}")
        return

    try:
        while not st.stop:
            frame = None
            with cast(threading.Lock, st.lock):
                if st.frame is not None:
                    frame   = st.frame
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
    control_mode.start()

    show_mask       = config.SHOW_DEBUG_WINDOWS
    fps_timer       = time.perf_counter()
    fps_count       = 0
    headless        = False
    cam_label       = "reCamera USB"

    state  = _WorkerState(lock=threading.Lock())
    worker = threading.Thread(
        target=_inference_worker, args=(state,), daemon=True, name="inference-worker"
    )
    worker.start()

    capture = Capture(UVC_DEVICE, cam_label)

    print(f"\nCamera engine started (UVC direct, MJPG).")
    print(f"Inference at {INFER_SCALE:.0%} res every {DETECT_EVERY} display frames.")
    print("Keys: q=quit  d=toggle mask  m=toggle manual/auto\n")

    last_read_ms = 0.0
    last_proc_ms = 0.0

    try:
        while True:
            t_read       = time.perf_counter()
            frame        = capture.read()
            last_read_ms = (time.perf_counter() - t_read) * 1000.0

            if frame is None:
                time.sleep(0.005)
                continue

            # ------------------------------------------------------------------
            # MANUAL MODE
            # ------------------------------------------------------------------
            if control_mode.is_manual():
                h, w  = frame.shape[:2]
                display = (
                    cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    if (w, h) != (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    else frame.copy()
                )
                _draw_camera_badge(display, cam_label)
                _draw_mode_badge(display)

                try:
                    web_server.push_frame(display)
                except Exception as e:
                    print(f"Warning: push_frame failed: {e}")

                if not headless:
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        if key == ord("m"):
                            print(f"[main] Mode: {control_mode.get_mode()} (gestuurd door controller)")
                    except Exception as e:
                        print(f"GUI unavailable (headless): {e}")
                        headless = True

                time.sleep(0.005)
                continue

            # ------------------------------------------------------------------
            # AUTONOMOUS MODE
            # ------------------------------------------------------------------
            with cast(threading.Lock, state.lock):
                state.frame = frame

            with cast(threading.Lock, state.lock):
                res = state.result

            if res is None:
                continue

            annotated, _, debug, mask, last_proc_ms = res
            if annotated is None:
                continue

            h, w = annotated.shape[:2]
            display = (
                cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                if (w, h) != (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                else annotated
            )
            display = np.asarray(display)
            _draw_camera_badge(display, cam_label)
            _draw_mode_badge(display)

            try:
                web_server.push_frame(display)
            except Exception as e:
                print(f"Warning: push_frame failed: {e}")

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
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d"):
                        show_mask = not show_mask
                        if not show_mask:
                            cv2.destroyWindow("CV Mask")
                    if key == ord("m"):
                        new = control_mode.toggle()
                        print(f"[main] Mode switched to: {new}")
                except Exception as e:
                    print(f"GUI unavailable (headless): {e}")
                    headless = True

    finally:
        state.stop = True
        control_mode.stop()
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
        capture.release()
        cv2.destroyAllWindows()
        if ON_PI:
            manual_controller.stop()
            gripper.shutdown()
            arm.shutdown()
            pivot.shutdown()
            lift.shutdown()
            turntable.shutdown()
            motor.shutdown()


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

    fusion   = FusionEngine()
    debug    = {"ai_count": 0, "cv_count": 0, "fused_count": 0, "confirmed_count": 0, "possible_count": 0}
    possible = None

    try:
        try:
            fusion.process_frame(np.asarray(frame))  # warmup
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
"""Lightweight web dashboard voor de Strawberry Fusion Detector.

Publieke API
-----------
start(host, port)   – start Flask in een daemon-thread (eenmalig)
push_frame(bgr)     – voed een geannoteerd BGR numpy-array aan de MJPEG-stream
push_log(line)      – stuur een logregel naar alle SSE-clients

REST API (JSON)
---------------
GET  /api/cv_config       – huidige CVConfig als JSON
POST /api/cv_config       – update CVConfig velden (partial update ok)
GET  /api/ai_enabled      – {"enabled": true/false}
POST /api/ai_enabled      – {"enabled": true/false}
GET  /api/cv_enabled      – {"enabled": true/false}
POST /api/cv_enabled      – {"enabled": true/false}
POST /api/cv_preset       – {"preset": "red"|"green"|"yellow"|"blue"|"orange"|"custom"}
GET  /api/full_config     – alle config.py knobs als JSON
POST /api/full_config     – partial update van config.py knobs (setattr live)
GET  /api/corner_sensors  – latest AS5600 reading per motor
GET  /api/dead_reckoning  – dead-reckoning position accumulators per motor
GET  /api/auto_move       – {"enabled": true/false} (arm extension + auto-grip)
POST /api/auto_move       – {"enabled": true/false}
"""
from __future__ import annotations

import io
import logging
import queue
import socket
import sys
import threading
import time
from typing import List, cast

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

logging.getLogger("werkzeug").setLevel(logging.ERROR)

_app = Flask(__name__)

_frame_lock:  threading.Lock    = threading.Lock()
_latest_jpeg: bytes | None      = None
_subs_lock:   threading.Lock    = threading.Lock()
_subs:        List[queue.Queue] = []
_started    = False
_start_lock = threading.Lock()

# =============================================================================
# COLOUR PRESETS
# =============================================================================

_PRESETS = {
    "red":    dict(h1_low=0,   h1_high=10,  h2_low=160, h2_high=179, sat_min=80,  val_min=50,  val_max=240),
    "orange": dict(h1_low=8,   h1_high=25,  h2_low=8,   h2_high=8,   sat_min=100, val_min=60,  val_max=240),
    "yellow": dict(h1_low=22,  h1_high=38,  h2_low=22,   h2_high=22,   sat_min=80,  val_min=60,  val_max=240),
    "green":  dict(h1_low=35,  h1_high=85,  h2_low=35,   h2_high=35,   sat_min=60,  val_min=40,  val_max=240),
    "blue":   dict(h1_low=100, h1_high=130, h2_low=100,   h2_high=100,   sat_min=60,  val_min=40,  val_max=240),
    "white":  dict(h1_low=0,   h1_high=179, h2_low=0,     h2_high=179,   sat_min=10,   val_min=180, val_max=255),
}

# =============================================================================
# FULL CONFIG SCHEMA
# =============================================================================

_FULL_CONFIG_SCHEMA = [
    ("YOLO_BASE_THRESHOLD",          float, 0.0, 1.0,   "YOLO base",            "thresholds", 0.5),
    ("CV_BASE_THRESHOLD",            float, 0.0, 1.0,   "CV base",              "thresholds", 0.4),
    ("CV_DIRECT_ACCEPT_THRESHOLD",   float, 0.0, 1.0,   "CV direct-accept",     "thresholds", 0.75),
    ("HIGH_AI_CONFIDENCE",           float, 0.0, 1.0,   "AI high conf",         "thresholds", 0.75),
    ("LOW_AI_CONFIDENCE",            float, 0.0, 1.0,   "AI low conf",          "thresholds", 0.35),
    ("YOLO_FUSION_WEIGHT",           float, 0.0, 1.0,   "YOLO weight",          "fusion", 0.6),
    ("CV_FUSION_WEIGHT",             float, 0.0, 1.0,   "CV weight",            "fusion", 0.4),
    ("CV_WEIGHT_REDNESS",            float, 0.0, 1.0,   "CV: redness",          "fusion", 0.4),
    ("CV_WEIGHT_CIRCULARITY",        float, 0.0, 1.0,   "CV: circularity",      "fusion", 0.3),
    ("CV_WEIGHT_SIZE",               float, 0.0, 1.0,   "CV: size",             "fusion", 0.2),
    ("CV_WEIGHT_TEXTURE",            float, 0.0, 1.0,   "CV: texture",          "fusion", 0.1),
    ("PERSISTENCE_REQUIRED",         int,   1,   20,    "Confirm frames (fused)","tracking", 3),
    ("PERSISTENCE_REQUIRED_CV_ONLY", int,   1,   20,    "Confirm frames (CV)",  "tracking", 5),
    ("PERSISTENCE_DECAY",            float, 0.0, 1.0,   "Conf decay / miss",    "tracking", 0.3),
    ("IOU_MATCH_THRESHOLD",          float, 0.0, 1.0,   "IoU match",            "tracking", 0.3),
    ("POSSIBLE_HIT_MIN_CONF",        float, 0.0, 1.0,   "Possible conf (fused)","tracking", 0.25),
    ("POSSIBLE_HIT_MIN_SEEN",        int,   1,   10,    "Possible seen (fused)","tracking", 2),
    ("POSSIBLE_CV_ONLY_MIN_CONF",    float, 0.0, 1.0,   "Possible conf (CV)",   "tracking", 0.3),
    ("POSSIBLE_CV_ONLY_MIN_SEEN",    int,   1,   10,    "Possible seen (CV)",   "tracking", 3),
    ("POSSIBLE_AI_ONLY_MIN_CONF",    float, 0.0, 1.0,   "Possible conf (AI)",   "tracking", 0.3),
    ("POSSIBLE_AI_ONLY_MIN_SEEN",    int,   1,   10,    "Possible seen (AI)",   "tracking", 2),
    ("POSSIBLE_AI_CONF_WEIGHT",      float, 0.0, 1.0,   "AI conf weight",       "tracking", 0.6),
    ("POSSIBLE_TARGET_MIN_CONF",     float, 0.0, 1.0,   "Target min conf",      "tracking", 0.4),
    ("MIN_CONTOUR_AREA",             int,   10,  5000,  "Min contour (px²)",    "shape", 300),
    ("CONVEXITY_MIN_AREA",           int,   100, 20000, "Watershed split (px²)","shape", 2000),
    ("MAX_RECHECKS",                 int,   0,   10,    "Max rechecks",         "zoom", 2),
    ("ZOOM_SCALE_FACTOR",            float, 1.0, 4.0,   "Scale factor",         "zoom", 2.0),
    ("RECHECK_AI_CONF",              float, 0.0, 1.0,   "Recheck AI thresh",    "zoom", 0.4),
    ("RECHECK_CV_CONF",              float, 0.0, 1.0,   "Recheck CV thresh",    "zoom", 0.35),
]


def _read_full_config() -> dict:
    import config as _cfg
    return {k: t(getattr(_cfg, k)) for k, t, *_ in _FULL_CONFIG_SCHEMA if getattr(_cfg, k, None) is not None}


def _write_full_config(data: dict) -> dict:
    import config as _cfg
    changed = {}
    for key, typ, mn, mx, *_ in _FULL_CONFIG_SCHEMA:
        if key not in data:
            continue
        try:
            val = max(typ(mn), min(typ(mx), typ(data[key])))
            setattr(_cfg, key, val)
            changed[key] = val
        except (ValueError, TypeError):
            pass
    return changed


def _get_schema_defaults() -> dict:
    return {e[0]: e[1](e[6]) for e in _FULL_CONFIG_SCHEMA}


# =============================================================================
# PUBLIC API
# =============================================================================

def push_frame(bgr: np.ndarray) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if ok:
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


def push_log(line: str) -> None:
    with _subs_lock:
        dead = [q for q in _subs if q.full()]
        for q in dead:
            _subs.remove(q)
        for q in _subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "???.???.???.???"


def start(host: str = "0.0.0.0", port: int = 8080) -> None:
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    sys.stdout = _Tee(cast(io.TextIOBase, sys.stdout))
    t = threading.Thread(
        target=lambda: _app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True, name="flask-dashboard",
    )
    t.start()
    time.sleep(0.4)
    print(f"Dashboard: http://{_local_ip()}:{port}/")


# =============================================================================
# STDOUT INTERCEPTOR
# =============================================================================

class _Tee(io.TextIOBase):
    def __init__(self, wrapped: io.TextIOBase) -> None:
        self._w = wrapped
        self._buf = ""

    def write(self, s: str) -> int:
        self._w.write(s)
        self._w.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                push_log(line)
        return len(s)

    def flush(self) -> None: self._w.flush()
    def fileno(self) -> int: return self._w.fileno()
    def isatty(self) -> bool:
        try: return self._w.isatty()
        except Exception: return False


# =============================================================================
# MJPEG
# =============================================================================

def _gen_mjpeg():
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / 20.0)


# =============================================================================
# FLASK ROUTES
# =============================================================================

_NO_CACHE = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache", "Expires": "0", "X-Accel-Buffering": "no",
}


@_app.route("/video_feed")
def route_video():
    resp = Response(_gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
    for k, v in _NO_CACHE.items():
        resp.headers[k] = v
    return resp


@_app.route("/snapshot")
def route_snapshot():
    with _frame_lock:
        jpeg = _latest_jpeg
    if jpeg is None:
        return Response(status=204)
    resp = Response(jpeg, mimetype="image/jpeg")
    for k, v in _NO_CACHE.items():
        resp.headers[k] = v
    return resp


@_app.route("/logs")
def route_logs():
    def sse_stream():
        q: queue.Queue = queue.Queue(maxsize=256)
        with _subs_lock:
            _subs.append(q)
        try:
            while True:
                try:
                    line = q.get(timeout=20)
                    safe = line.replace("\r", "").replace("\n", " ")
                    yield f"data: {safe}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _subs_lock:
                if q in _subs:
                    _subs.remove(q)
    return Response(sse_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── CV config ────────────────────────────────────────────────────────────────

@_app.route("/api/cv_config", methods=["GET"])
def api_cv_config_get():
    from detection import get_cv_config
    return jsonify(get_cv_config().to_dict())


@_app.route("/api/cv_config", methods=["POST"])
def api_cv_config_post():
    from detection import get_cv_config, set_cv_config, CVConfig
    data = request.get_json(force=True, silent=True) or {}
    new_cfg = CVConfig.from_dict({**get_cv_config().to_dict(), **data})
    set_cv_config(new_cfg)
    print(f"[WebUI] CV config updated: {new_cfg.to_dict()}")
    return jsonify({"ok": True, "config": new_cfg.to_dict()})


@_app.route("/api/cv_preset", methods=["POST"])
def api_cv_preset():
    from detection import get_cv_config, set_cv_config, CVConfig
    data = request.get_json(force=True, silent=True) or {}
    preset = data.get("preset", "").lower()
    if preset not in _PRESETS:
        return jsonify({"ok": False, "error": f"Unknown preset '{preset}'. Valid: {list(_PRESETS)}"}), 400
    new_cfg = CVConfig.from_dict({**get_cv_config().to_dict(), **_PRESETS[preset]})
    set_cv_config(new_cfg)
    print(f"[WebUI] CV preset applied: {preset}")
    return jsonify({"ok": True, "preset": preset, "config": new_cfg.to_dict()})


# ── Full config ───────────────────────────────────────────────────────────────

@_app.route("/api/full_config", methods=["GET"])
def api_full_config_get():
    return jsonify(_read_full_config())


@_app.route("/api/full_config", methods=["POST"])
def api_full_config_post():
    data = request.get_json(force=True, silent=True) or {}
    changed = _write_full_config(data)
    print(f"[WebUI] Full config updated: {changed}")
    return jsonify({"ok": True, "changed": changed})


@_app.route("/api/default_config", methods=["GET"])
def api_default_config_get():
    return jsonify(_get_schema_defaults())


# ── AI toggle ─────────────────────────────────────────────────────────────────

@_app.route("/api/ai_enabled", methods=["GET"])
def api_ai_get():
    from fusion_engine import is_ai_enabled
    return jsonify({"enabled": is_ai_enabled()})


@_app.route("/api/ai_enabled", methods=["POST"])
def api_ai_post():
    from fusion_engine import set_ai_enabled
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    set_ai_enabled(enabled)
    return jsonify({"ok": True, "enabled": enabled})


# ── CV toggle ─────────────────────────────────────────────────────────────────

@_app.route("/api/cv_enabled", methods=["GET"])
def api_cv_get():
    from fusion_engine import is_cv_enabled
    return jsonify({"enabled": is_cv_enabled()})


@_app.route("/api/cv_enabled", methods=["POST"])
def api_cv_post():
    from fusion_engine import set_cv_enabled
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    set_cv_enabled(enabled)
    return jsonify({"ok": True, "enabled": enabled})


# ── Corner sensors ────────────────────────────────────────────────────────────

@_app.route("/api/corner_sensors")
def api_corner_sensors():
    import importlib
    _MOTORS = [
        ("turntable", "turntable", False),
        ("lift",      "lift",      True),
        ("gripper",   "gripper",   False),
        ("arm",       "arm",       False),
        ("pivot",     "pivot",     False),
    ]
    result = {}
    for name, mod_name, dual in _MOTORS:
        try:
            mod = importlib.import_module(mod_name)
            if dual:
                readings = mod.get_sensor_readings()
                result[name] = {"available": any(v is not None for v in readings.values()), "channels": readings}
            else:
                r = mod.get_sensor_reading()
                result[name] = {"available": r is not None, "data": r}
        except Exception as e:
            result[name] = {"available": False, "error": str(e)}
    return jsonify(result)


# ── Dead-reckoning positions ──────────────────────────────────────────────────

@_app.route("/api/dead_reckoning")
def api_dead_reckoning():
    """
    Returns the current dead-reckoning accumulator state for every motor
    that tracks open-loop position.  When a sensor IS present the motor
    still maintains the accumulator as a fallback, so both are returned.

    Response shape per motor:
        dead_pos      – raw accumulator value (speed-units × seconds)
        estimated_deg – dead_pos × SPEED_TO_DEG  (approximate degrees)
        speed_to_deg  – the calibration constant used for conversion
        has_sensor    – true when an AS5600 is wired and detected
        zero_deg      – sensor reading captured at init() (null if no sensor)
        min_deg       – soft travel limit (null if unlimited)
        max_deg       – soft travel limit (null if unlimited)
    """
    import importlib
    _MOTORS_DR = [
        ("turntable", "turntable"),
        ("lift",      "lift"),
        ("arm",       "arm"),
        ("pivot",     "pivot"),
    ]
    result = {}
    for name, mod_name in _MOTORS_DR:
        try:
            mod = importlib.import_module(mod_name)
            dead_pos   = getattr(mod, "_dead_pos",    [0.0])[0]
            s2d        = getattr(mod, "SPEED_TO_DEG", None)
            has_sensor = getattr(mod, "_sensor_mgr",  None) is not None
            zero_deg   = getattr(mod, "_zero_deg",    None)
            min_deg    = getattr(mod, "MIN_DEG",      None)
            max_deg    = getattr(mod, "MAX_DEG",      None)
            est_deg    = round(dead_pos * s2d, 2) if s2d is not None else None
            result[name] = {
                "dead_pos":      round(dead_pos, 3),
                "estimated_deg": est_deg,
                "speed_to_deg":  s2d,
                "has_sensor":    has_sensor,
                "zero_deg":      round(zero_deg, 2) if zero_deg is not None else None,
                "min_deg":       min_deg,
                "max_deg":       max_deg,
            }
        except Exception as e:
            result[name] = {"error": str(e)}
    return jsonify(result)


# ── Servo status ──────────────────────────────────────────────────────────────

@_app.route("/api/servo_status")
def api_servo_status():
    import servo_status
    states = servo_status.get_all()
    result = []
    for s in states:
        result.append({
            "id":        s.id,
            "name":      s.name,
            "status":    s.status,
            "speed":     s.speed,
            "simulated": s.simulated,
        })
    return jsonify(result)

# ── Home / Kill switch ────────────────────────────────────────────────────────

_kill_active = False
_kill_lock   = threading.Lock()

def is_killed() -> bool:
    with _kill_lock:
        return _kill_active

@_app.route("/api/home", methods=["POST"])
def api_home():
    try:
        import motor
        threading.Thread(target=motor.home_all, daemon=True, name="home-trigger").start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@_app.route("/api/kill", methods=["POST"])
def api_kill():
    global _kill_active
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    if enabled:
        print("[WebUI] Kill switch: ACTIVE — shutting down process")
        def _do_kill():
            time.sleep(0.3)
            import os, signal
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_do_kill, daemon=True).start()
    with _kill_lock:
        _kill_active = enabled
    return jsonify({"ok": True, "kill_active": enabled})

# ── Auto-move toggle (arm + gripper) ──────────────────────────────────────────

@_app.route("/api/auto_move", methods=["GET"])
def api_auto_move_get():
    import config
    return jsonify({"enabled": getattr(config, "AUTO_MODE_ALLOW_MOVE", True)})

@_app.route("/api/auto_move", methods=["POST"])
def api_auto_move_post():
    import config
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    setattr(config, "AUTO_MODE_ALLOW_MOVE", enabled)
    print(f"[WebUI] Auto-move set to: {enabled} (arm + gripper)")
    return jsonify({"ok": True, "enabled": enabled})

@_app.route("/favicon.ico")
def route_favicon():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="28">&#x1F353;</text></svg>'
    return Response(svg, mimetype="image/svg+xml")


@_app.route("/")
def route_index():
    return _HTML


# =============================================================================
# DASHBOARD HTML
# =============================================================================

_HTML = r"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🍓 Strawberry Detector</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c10;--bg2:#0d1219;--sur:#111720;--brd:#1e2a38;--brd2:#243040;
  --acc:#ff3d5a;--grn:#2ddb72;--grn-lo:rgba(45,219,114,.12);
  --yel:#f5c842;--blu:#3da9f5;--ora:#ff8c42;--pur:#b06cff;
  --txt:#c8d6e8;--mut:#5a7080;
  --mono:'IBM Plex Mono',monospace;--sans:'Space Grotesk',system-ui,sans-serif;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--txt);font-family:var(--mono);font-size:12px}
body{display:flex;flex-direction:column}

/* header */
header{display:flex;align-items:center;gap:12px;padding:0 16px;height:48px;background:var(--sur);border-bottom:1px solid var(--brd);flex-shrink:0}
.logo{font-size:18px}
h1{font-family:var(--sans);font-size:13px;font-weight:700;white-space:nowrap}
h1 span{color:var(--acc)}
.sep{width:1px;height:24px;background:var(--brd);flex-shrink:0}
.pill{display:flex;align-items:center;gap:5px;padding:2px 9px;border-radius:100px;border:1px solid var(--brd);background:var(--bg2);font-size:10px;color:var(--mut)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--mut);flex-shrink:0}
.dot.on{background:var(--grn);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.stats{display:flex;gap:5px;margin-left:auto;align-items:center;flex-wrap:wrap}
.st{display:flex;flex-direction:column;align-items:center;padding:3px 9px;background:var(--bg2);border:1px solid var(--brd);border-radius:6px;min-width:44px}
.st-l{font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:.7px}
.st-v{font-family:var(--sans);font-size:15px;font-weight:700;line-height:1.3}
#s-fps{color:var(--grn)}#s-hits{color:var(--acc)}#s-cam{font-size:11px;font-family:var(--mono)}

/* main layout */
main{flex:1;display:flex;min-height:0}

/* settings panel */
.panel{width:280px;min-width:200px;max-width:55vw;flex-shrink:0;display:flex;flex-direction:column;background:var(--sur);border-right:1px solid var(--brd);min-height:0}

/* tabs */
.tabs{display:flex;flex-shrink:0;border-bottom:1px solid var(--brd);overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tab{flex-shrink:0;padding:0 12px;height:32px;cursor:pointer;font-family:var(--mono);font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:.8px;background:none;border:none;border-bottom:2px solid transparent;transition:color .15s,border-color .15s}
.tab:hover{color:var(--txt)}
.tab.on{color:var(--txt);border-bottom-color:var(--acc)}
.pane{display:none;flex:1;overflow-y:auto}
.pane.on{display:block}
.pane::-webkit-scrollbar{width:3px}
.pane::-webkit-scrollbar-thumb{background:var(--brd2);border-radius:2px}
.sec{padding:10px;border-bottom:1px solid var(--brd)}
.sec-t{font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}

/* toggle rows */
.ai-row{display:flex;align-items:center;justify-content:space-between;padding:7px 9px;border-radius:6px;border:1px solid var(--brd);background:var(--bg2);margin-bottom:6px}
.ai-row:last-child{margin-bottom:0}
.ai-lbl small{display:block;font-size:9px;color:var(--mut);margin-top:1px}
.sw{position:relative;width:34px;height:19px;flex-shrink:0}
.sw input{opacity:0;width:0;height:0}
.sw-t{position:absolute;inset:0;border-radius:10px;background:var(--brd2);cursor:pointer;transition:background .2s}
.sw-t::after{content:'';position:absolute;width:13px;height:13px;border-radius:50%;background:var(--mut);top:3px;left:3px;transition:all .2s}
.sw input:checked+.sw-t{background:var(--grn)}
.sw input:checked+.sw-t::after{background:#fff;transform:translateX(15px)}

/* combined toggle button */
.combo-btn{width:100%;padding:8px 12px;font-family:var(--mono);font-size:11px;font-weight:600;border-radius:6px;cursor:pointer;transition:all .15s;letter-spacing:.4px;display:flex;align-items:center;justify-content:space-between;margin-top:2px}
.combo-btn .cb-lbl small{display:block;font-size:9px;font-weight:400;margin-top:1px;text-align:left}
.combo-btn .cb-state{font-size:10px;padding:2px 8px;border-radius:10px;border:1px solid currentColor;opacity:.8}
.combo-both-on{border:1px solid var(--grn);background:var(--grn-lo);color:var(--grn)}
.combo-mixed{border:1px solid var(--yel);background:rgba(245,200,66,.08);color:var(--yel)}
.combo-both-off{border:1px solid var(--brd2);background:var(--bg2);color:var(--mut)}

/* presets */
.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.pbtn{font-family:var(--mono);font-size:10px;padding:5px 3px;border:1px solid var(--brd);border-radius:5px;background:var(--bg2);color:var(--mut);cursor:pointer;text-align:center;transition:all .15s}
.pbtn:hover{background:var(--brd2);color:var(--txt)}
.pbtn.on{border-color:var(--acc);color:var(--acc);background:rgba(255,61,90,.08)}
.pdot{display:block;width:9px;height:9px;border-radius:50%;margin:0 auto 2px}

/* sliders */
.row{margin-bottom:7px}
.row label{display:flex;justify-content:space-between;font-size:10px;color:var(--mut);margin-bottom:2px}
.row label span{color:var(--txt);font-weight:600}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:2px;outline:none;cursor:pointer;background:var(--brd2)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--blu);cursor:pointer}
input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;border:none;background:var(--blu);cursor:pointer}
.ora-thumb::-webkit-slider-thumb{background:var(--ora)}
.ora-thumb::-moz-range-thumb{background:var(--ora)}

/* hue band */
.hband{height:7px;border-radius:3px;margin-bottom:8px;background:linear-gradient(to right,hsl(0,100%,50%),hsl(60,100%,50%),hsl(120,100%,50%),hsl(180,100%,50%),hsl(240,100%,50%),hsl(300,100%,50%),hsl(360,100%,50%));position:relative}
.hmark{position:absolute;top:-2px;width:3px;height:11px;background:#fff;border-radius:2px;transform:translateX(-50%);pointer-events:none;transition:left .1s}

/* buttons */
.btn-row{display:flex;gap:5px;margin-top:8px}
.btn{flex:1;padding:6px;font-family:var(--mono);font-size:11px;font-weight:600;border-radius:5px;cursor:pointer;transition:all .15s;letter-spacing:.4px}
.btn-g{border:1px solid var(--grn);background:var(--grn-lo);color:var(--grn)}
.btn-g:hover{background:rgba(45,219,114,.22)}
.btn-o{border:1px solid var(--ora);background:rgba(255,140,66,.1);color:var(--ora)}
.btn-o:hover{background:rgba(255,140,66,.22)}
.btn-r{border:1px solid var(--brd2);background:var(--bg2);color:var(--mut)}
.btn-r:hover{border-color:var(--yel)!important;color:var(--yel)!important;background:rgba(245,200,66,.08)!important}
.btn:active{transform:scale(.97)}
.btn.busy{opacity:.5;pointer-events:none}
.fb{font-size:9px;text-align:center;margin-top:4px;height:11px;color:var(--mut)}
.fb.ok{color:var(--grn)}.fb.err{color:var(--acc)}

/* knob rows */
.krow{margin-bottom:8px}
.krow label{display:flex;justify-content:space-between;align-items:baseline;font-size:10px;color:var(--mut);margin-bottom:2px}
.krow .kn{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.krow .kv{color:var(--txt);font-weight:600;min-width:32px;text-align:right}
input[type=number]{width:100%;padding:2px 5px;background:var(--bg2);border:1px solid var(--brd);border-radius:4px;color:var(--txt);font-family:var(--mono);font-size:11px;outline:none}
input[type=number]:focus{border-color:var(--ora)}

/* group sub-header */
.grp-h{font-size:9px;color:var(--blu);text-transform:uppercase;letter-spacing:.7px;margin:10px 0 5px;padding-top:8px;border-top:1px solid var(--brd)}
.grp-h:first-child{border-top:none;margin-top:0;padding-top:0}

/* sensor cards */
.scard{background:var(--bg2);border:1px solid var(--brd);border-radius:6px;padding:7px 9px;margin-bottom:5px}
.scard.live{border-color:var(--grn)}
.scard.dr{border-color:var(--pur)}
.scard.dead{opacity:.55}
.scard-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.sname{font-family:var(--sans);font-size:11px;font-weight:700}
.sbadge{font-size:9px;padding:1px 6px;border-radius:8px;background:var(--brd2);color:var(--mut)}
.sbadge.on{background:var(--grn-lo);color:var(--grn)}
.sbadge.dr{background:rgba(176,108,255,.15);color:var(--pur)}
.srow{display:flex;justify-content:space-between;font-size:10px;margin-top:2px}
.sk{color:var(--mut)}.sv{color:var(--txt);font-weight:600}
.sarc{height:4px;border-radius:2px;margin-top:5px;background:var(--brd2);position:relative;overflow:hidden}
.sarc-f{position:absolute;top:0;left:0;height:100%;background:var(--grn);border-radius:2px;transition:width .25s}
.sarc-f.dr{background:var(--pur)}
/* DR position bar uses min/max range; neutral (zero) centred */
.sarc-dr{height:4px;border-radius:2px;margin-top:5px;background:var(--brd2);position:relative;overflow:hidden}
.sarc-dr-f{position:absolute;top:0;height:100%;background:var(--pur);border-radius:2px;transition:left .2s,width .2s}
/* zero tick */
.sarc-dr::after{content:'';position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.2);left:50%}
.dr-section{margin-top:6px;padding-top:6px;border-top:1px dashed var(--brd2)}
.dr-label{font-size:9px;color:var(--pur);text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px}

/* drag divider */
.div{width:4px;flex-shrink:0;background:var(--brd);cursor:col-resize;transition:background .15s;z-index:10}
.div:hover,.div.drag{background:var(--acc)}

/* video panel */
.video{flex:1;min-width:0;position:relative;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}
.video::after{content:'';position:absolute;inset:0;background:repeating-linear-gradient(to bottom,transparent 0,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);pointer-events:none;z-index:2}
#feed{max-width:100%;max-height:100%;object-fit:contain;display:none;z-index:1}
.nosig{display:flex;flex-direction:column;align-items:center;gap:10px;color:var(--mut);z-index:3;pointer-events:none}
.nosig-ico{font-size:48px;opacity:.15}
.nosig p{font-size:11px}
.clabel{position:absolute;z-index:3;font-size:9px;color:rgba(255,255,255,.22);letter-spacing:.5px;text-transform:uppercase}
.clabel.tl{top:7px;left:9px}.clabel.tr{top:7px;right:9px}.clabel.bl{bottom:7px;left:9px}

/* console panel */
.con{width:320px;min-width:160px;max-width:65vw;flex-shrink:0;display:flex;flex-direction:column;background:var(--sur);min-height:0}
.con-bar{display:flex;align-items:center;justify-content:space-between;padding:0 9px;height:32px;border-bottom:1px solid var(--brd);flex-shrink:0}
.con-t{font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:1px}
.btns{display:flex;gap:4px}
button{font-family:var(--mono);font-size:10px;padding:2px 8px;border:1px solid var(--brd);border-radius:4px;background:var(--bg2);color:var(--mut);cursor:pointer;transition:all .15s}
button:hover{background:var(--brd2);color:var(--txt)}
button.on{border-color:var(--grn);color:var(--grn);background:var(--grn-lo)}
#log{flex:1;overflow-y:auto;padding:5px 4px 5px 9px;min-height:0}
#log::-webkit-scrollbar{width:3px}
#log::-webkit-scrollbar-thumb{background:var(--brd2);border-radius:2px}
.ll{display:grid;grid-template-columns:50px 1fr;gap:5px;line-height:1.6;white-space:pre-wrap;word-break:break-all}
.lt{font-size:9px;color:#2a3a48;padding-top:3px}
.lm{color:var(--txt)}
.lm.fps{color:var(--grn)}.lm.err{color:#ff5370}.lm.warn{color:var(--yel)}.lm.info{color:var(--blu)}.lm.ok{color:var(--grn);font-weight:600}.lm.sep{color:var(--brd2)}
.rbadge{display:inline-block;margin-left:6px;font-size:9px;padding:0 5px;border-radius:8px;background:var(--brd2);color:var(--mut);vertical-align:middle;line-height:15px}

/* footer */
footer{display:flex;align-items:center;justify-content:space-between;padding:0 12px;height:26px;background:var(--sur);border-top:1px solid var(--brd);font-size:10px;color:var(--mut);flex-shrink:0}
#fc{display:flex;align-items:center;gap:5px}

@media(max-width:900px){main{flex-direction:column}.video{flex:none;height:38vh}.div{display:none}.panel{width:100%!important;min-width:unset;max-height:260px;border-right:none;border-bottom:1px solid var(--brd)}.con{width:100%!important;min-width:unset}.stats{display:none}}
</style>
</head>
<body>

<header>
  <div class="logo">🍓</div>
  <h1>Strawberry <span>Detector</span></h1>
  <div class="sep"></div>
  <div class="pill">
    <div class="dot" id="liveDot"></div>
    <span id="liveText">Verbinden…</span>
  </div>
  <div class="stats">
    <div class="st"><span class="st-l">FPS</span><span class="st-v" id="s-fps">—</span></div>
    <div class="st"><span class="st-l">AI</span><span class="st-v" id="s-ai">—</span></div>
    <div class="st"><span class="st-l">CV</span><span class="st-v" id="s-cv">—</span></div>
    <div class="st"><span class="st-l">Fused</span><span class="st-v" id="s-fused">—</span></div>
    <div class="st"><span class="st-l">Hits</span><span class="st-v" id="s-hits">—</span></div>
    <div class="st"><span class="st-l">Poss</span><span class="st-v" id="s-poss">—</span></div>
    <div class="st"><span class="st-l">Cam</span><span class="st-v" id="s-cam">—</span></div>
    <div class="sep"></div>
    <button id="homeBtn" onclick="triggerHome()" style="padding:3px 10px;border-color:var(--yel);color:var(--yel);background:rgba(245,200,66,.08)">🏠 Home</button>
    <button id="killBtn" onclick="toggleKill()" style="padding:3px 10px;border-color:var(--acc);color:var(--acc);background:rgba(255,61,90,.08)">☠ KILL</button>
  </div>
</header>

<main>

<!-- ── SETTINGS PANEL ── -->
<div class="panel" id="settingsPanel">
  <div class="tabs">
    <button class="tab on" onclick="switchTab('detect')">Detect</button>
    <button class="tab" onclick="switchTab('thresholds')">Thresh</button>
    <button class="tab" onclick="switchTab('tracking')">Track</button>
    <button class="tab" onclick="switchTab('advanced')">Advanced</button>
    <button class="tab" onclick="switchTab('sensors')">Sensors</button>
  </div>

  <!-- ── DETECT ── -->
  <div class="pane on" id="tab-detect">
    <div class="sec">
      <div class="sec-t">Detector mode</div>
      <div class="ai-row">
        <div class="ai-lbl">AI (YOLO)<small id="aiSub">Loading…</small></div>
        <label class="sw"><input type="checkbox" id="aiToggle" onchange="onAiToggle(this.checked)"><span class="sw-t"></span></label>
      </div>
      <div class="ai-row">
        <div class="ai-lbl">CV (OpenCV)<small id="cvSub">Loading…</small></div>
        <label class="sw"><input type="checkbox" id="cvToggle" onchange="onCvToggle(this.checked)"><span class="sw-t"></span></label>
      </div>
      <button class="combo-btn combo-both-off" id="comboBtn" onclick="onComboToggle()">
        <div class="cb-lbl">AI + CV<small id="comboSub">Both off</small></div>
        <span class="cb-state" id="comboState">OFF</span>
      </button>
    </div>
    <div class="sec">
      <div class="sec-t">Autonomous mode</div>
      <div class="ai-row">
        <div class="ai-lbl">Allow movement<small>Arm extension + auto-grip</small></div>
        <label class="sw"><input type="checkbox" id="autoMoveToggle" onchange="onAutoMoveToggle(this.checked)"><span class="sw-t"></span></label>
      </div>
    </div>
    <div class="sec">
      <div class="sec-t">Colour preset</div>
      <div class="presets">
        <button class="pbtn on" data-preset="red" onclick="applyPreset('red')"><span class="pdot" style="background:#e03030"></span>Red</button>
        <button class="pbtn" data-preset="orange" onclick="applyPreset('orange')"><span class="pdot" style="background:#e07830"></span>Orange</button>
        <button class="pbtn" data-preset="yellow" onclick="applyPreset('yellow')"><span class="pdot" style="background:#d4c030"></span>Yellow</button>
        <button class="pbtn" data-preset="green" onclick="applyPreset('green')"><span class="pdot" style="background:#30c050"></span>Green</button>
        <button class="pbtn" data-preset="blue" onclick="applyPreset('blue')"><span class="pdot" style="background:#3070e0"></span>Blue</button>
        <button class="pbtn" data-preset="white" onclick="applyPreset('white')"><span class="pdot" style="background:#e8e8e8"></span>White</button>
      </div>
    </div>
    <div class="sec">
      <div class="sec-t">HSV fine-tune</div>
      <div style="font-size:9px;color:var(--mut);margin-bottom:3px">HUE BAND 1</div>
      <div class="hband"><div class="hmark" id="hm1l"></div><div class="hmark" id="hm1h" style="background:#adf"></div></div>
      <div class="row"><label>H1 low <span id="v-h1l">0</span></label><input type="range" min="0" max="179" value="0" id="sl-h1l" oninput="sliderChanged()"></div>
      <div class="row"><label>H1 high <span id="v-h1h">10</span></label><input type="range" min="0" max="179" value="10" id="sl-h1h" oninput="sliderChanged()"></div>
      <div style="font-size:9px;color:var(--mut);margin:5px 0 3px">HUE BAND 2</div>
      <div class="hband"><div class="hmark" id="hm2l"></div><div class="hmark" id="hm2h" style="background:#adf"></div></div>
      <div class="row"><label>H2 low <span id="v-h2l">160</span></label><input type="range" min="0" max="179" value="160" id="sl-h2l" oninput="sliderChanged()"></div>
      <div class="row"><label>H2 high <span id="v-h2h">179</span></label><input type="range" min="0" max="179" value="179" id="sl-h2h" oninput="sliderChanged()"></div>
      <div style="font-size:9px;color:var(--mut);margin:5px 0 3px">SAT / VAL</div>
      <div class="row"><label>Sat min <span id="v-sat">80</span></label><input type="range" min="0" max="255" value="80" id="sl-sat" oninput="sliderChanged()"></div>
      <div class="row"><label>Val min <span id="v-vmin">50</span></label><input type="range" min="0" max="255" value="50" id="sl-vmin" oninput="sliderChanged()"></div>
      <div class="row"><label>Val max <span id="v-vmax">240</span></label><input type="range" min="0" max="255" value="240" id="sl-vmax" oninput="sliderChanged()"></div>
      <div class="btn-row">
        <button class="btn btn-g" id="applyHSVBtn" onclick="applyHSV()">▶ Apply</button>
        <button class="btn btn-r" onclick="resetHSV()">↺ Reset</button>
      </div>
      <div class="fb" id="fb-hsv"></div>
    </div>
  </div>

  <!-- ── THRESHOLDS ── -->
  <div class="pane" id="tab-thresholds">
    <div class="sec">
      <div class="sec-t">Confidence thresholds</div>
      <div id="knobs-thresholds"></div>
      <div class="btn-row">
        <button class="btn btn-o" id="applyThresholds" onclick="applyGroup('thresholds')">▶ Apply</button>
        <button class="btn btn-r" onclick="resetGroup('thresholds')">↺ Reset</button>
      </div>
      <div class="fb" id="fb-thresholds"></div>
    </div>
  </div>

  <!-- ── TRACKING ── -->
  <div class="pane" id="tab-tracking">
    <div class="sec">
      <div class="sec-t">Persistence & possible-hit</div>
      <div id="knobs-tracking"></div>
      <div class="btn-row">
        <button class="btn btn-o" id="applyTracking" onclick="applyGroup('tracking')">▶ Apply</button>
        <button class="btn btn-r" onclick="resetGroup('tracking')">↺ Reset</button>
      </div>
      <div class="fb" id="fb-tracking"></div>
    </div>
  </div>

  <!-- ── ADVANCED ── -->
  <div class="pane" id="tab-advanced">
    <div class="sec">
      <div class="grp-h">Fusion weights</div>
      <div id="knobs-fusion"></div>
      <div class="grp-h">Shape / contour</div>
      <div id="knobs-shape"></div>
      <div class="grp-h">CVConfig shape</div>
      <div class="krow"><label><span class="kn">Circularity min</span><span class="kv" id="v-circ">0.55</span></label>
        <input type="range" min="0" max="1" step="0.01" value="0.55" id="sl-circ" oninput="document.getElementById('v-circ').textContent=parseFloat(this.value).toFixed(2)"></div>
      <div class="krow"><label><span class="kn">Max aspect ratio</span><span class="kv" id="v-asr">1.6</span></label>
        <input type="range" min="1" max="5" step="0.1" value="1.6" id="sl-asr" oninput="document.getElementById('v-asr').textContent=parseFloat(this.value).toFixed(1)"></div>
      <div class="krow"><label><span class="kn">Watershed FG thresh</span><span class="kv" id="v-wfg">0.35</span></label>
        <input type="range" min="0" max="1" step="0.01" value="0.35" id="sl-wfg" oninput="document.getElementById('v-wfg').textContent=parseFloat(this.value).toFixed(2)"></div>
      <div class="krow"><label><span class="kn">NMS IoU threshold</span><span class="kv" id="v-nms">0.35</span></label>
        <input type="range" min="0" max="1" step="0.01" value="0.35" id="sl-nms" oninput="document.getElementById('v-nms').textContent=parseFloat(this.value).toFixed(2)"></div>
      <div class="grp-h">Zoom recheck</div>
      <div id="knobs-zoom"></div>
      <div class="btn-row">
        <button class="btn btn-o" id="applyAdvanced" onclick="applyAdvanced()">▶ Apply all</button>
        <button class="btn btn-r" onclick="resetAdvanced()">↺ Reset</button>
      </div>
      <div class="fb" id="fb-advanced"></div>
    </div>
  </div>

  <!-- ── SENSORS ── -->
  <div class="pane" id="tab-sensors">
    <div class="sec">
      <div class="sec-t" style="display:flex;justify-content:space-between;align-items:center">
        Corner sensors (AS5600)
        <button onclick="refreshSensors()" style="padding:1px 7px;font-size:9px">⟳ Refresh</button>
      </div>
      <div id="sensorList"></div>
      <div style="font-size:9px;color:var(--mut);margin-top:8px">Auto-refreshes every 500 ms when active.</div>
    </div>
    <div class="sec">
      <div class="sec-t" style="display:flex;justify-content:space-between;align-items:center">
        Servo status
        <button onclick="refreshServos()" style="padding:1px 7px;font-size:9px">⟳ Refresh</button>
      </div>
      <div id="servoList"></div>
    </div>
  </div>
</div><!-- /panel -->

<div class="div" id="divLeft"></div>

<!-- ── VIDEO ── -->
<div class="video">
  <span class="clabel tl" id="camLabel">CAM</span>
  <span class="clabel tr" id="resLabel"></span>
  <span class="clabel bl" id="timeLabel"></span>
  <div class="nosig" id="noSignal"><div class="nosig-ico">📷</div><p>Wachten op videostream…</p></div>
  <img id="feed" alt="camera">
</div>

<div class="div" id="divRight"></div>

<!-- ── CONSOLE ── -->
<div class="con" id="consolePanel">
  <div class="con-bar">
    <span class="con-t">Console</span>
    <div class="btns">
      <button id="scrollBtn" class="on" onclick="toggleScroll()">↓ Auto</button>
      <button onclick="clearLog()">Wis</button>
    </div>
  </div>
  <div id="log"></div>
</div>

</main>

<footer>
  <div id="fc"><div class="dot" id="connDot"></div><span id="connText">Verbinden…</span></div>
  <span id="lc">0 regels</span>
</footer>

<script>
"use strict";
const MAX_LINES=600,RECONNECT_MS=3000;
let autoScroll=true,lineCount=0,evtSrc=null,lastText=null,lastCount=1,lastMsgEl=null;

const SCHEMA=[
  ["YOLO_BASE_THRESHOLD","float",0,1,"YOLO base","thresholds",0.5],
  ["CV_BASE_THRESHOLD","float",0,1,"CV base","thresholds",0.4],
  ["CV_DIRECT_ACCEPT_THRESHOLD","float",0,1,"CV direct-accept","thresholds",0.75],
  ["HIGH_AI_CONFIDENCE","float",0,1,"AI high conf","thresholds",0.75],
  ["LOW_AI_CONFIDENCE","float",0,1,"AI low conf","thresholds",0.35],
  ["YOLO_FUSION_WEIGHT","float",0,1,"YOLO weight","fusion",0.6],
  ["CV_FUSION_WEIGHT","float",0,1,"CV weight","fusion",0.4],
  ["CV_WEIGHT_REDNESS","float",0,1,"CV: redness","fusion",0.4],
  ["CV_WEIGHT_CIRCULARITY","float",0,1,"CV: circularity","fusion",0.3],
  ["CV_WEIGHT_SIZE","float",0,1,"CV: size","fusion",0.2],
  ["CV_WEIGHT_TEXTURE","float",0,1,"CV: texture","fusion",0.1],
  ["PERSISTENCE_REQUIRED","int",1,20,"Confirm frames (fused)","tracking",3],
  ["PERSISTENCE_REQUIRED_CV_ONLY","int",1,20,"Confirm frames (CV)","tracking",5],
  ["PERSISTENCE_DECAY","float",0,1,"Conf decay/miss","tracking",0.3],
  ["IOU_MATCH_THRESHOLD","float",0,1,"IoU match","tracking",0.3],
  ["POSSIBLE_HIT_MIN_CONF","float",0,1,"Possible conf (fused)","tracking",0.25],
  ["POSSIBLE_HIT_MIN_SEEN","int",1,10,"Possible seen (fused)","tracking",2],
  ["POSSIBLE_CV_ONLY_MIN_CONF","float",0,1,"Possible conf (CV)","tracking",0.3],
  ["POSSIBLE_CV_ONLY_MIN_SEEN","int",1,10,"Possible seen (CV)","tracking",3],
  ["POSSIBLE_AI_ONLY_MIN_CONF","float",0,1,"Possible conf (AI)","tracking",0.3],
  ["POSSIBLE_AI_ONLY_MIN_SEEN","int",1,10,"Possible seen (AI)","tracking",2],
  ["POSSIBLE_AI_CONF_WEIGHT","float",0,1,"AI conf weight","tracking",0.6],
  ["POSSIBLE_TARGET_MIN_CONF","float",0,1,"Target min conf","tracking",0.4],
  ["MIN_CONTOUR_AREA","int",10,5000,"Min contour (px²)","shape",300],
  ["CONVEXITY_MIN_AREA","int",100,20000,"Watershed split (px²)","shape",2000],
  ["MAX_RECHECKS","int",0,10,"Max rechecks","zoom",2],
  ["ZOOM_SCALE_FACTOR","float",1,4,"Scale factor","zoom",2.0],
  ["RECHECK_AI_CONF","float",0,1,"Recheck AI thresh","zoom",0.4],
  ["RECHECK_CV_CONF","float",0,1,"Recheck CV thresh","zoom",0.35],
];
const HSV_DEF={h1_low:0,h1_high:10,h2_low:160,h2_high:179,sat_min:80,val_min:50,val_max:240};
const SHP_DEF={contour_min_circularity:0.55,max_aspect_ratio:1.6,watershed_fg_thresh:0.35,nms_iou_threshold:0.35};
const _cfg={};

let _aiOn=true,_cvOn=true;

/* ── TABS ── */
let _sensorTimer=null;
function switchTab(name){
  document.querySelectorAll(".tab").forEach(b=>b.classList.remove("on"));
  document.querySelectorAll(".pane").forEach(p=>p.classList.remove("on"));
  document.querySelector(`.tab[onclick*="'${name}'"]`)?.classList.add("on");
  document.getElementById("tab-"+name)?.classList.add("on");
  clearInterval(_sensorTimer);_sensorTimer=null;
  if(name==="sensors"){
    refreshSensors();refreshServos();
    _sensorTimer=setInterval(()=>{refreshSensors();refreshServos();},500);
  }
}

/* ── KNOBS ── */
function buildKnobs(group){
  const c=document.getElementById("knobs-"+group);if(!c)return;c.innerHTML="";
  SCHEMA.filter(s=>s[5]===group).forEach(([key,typ,mn,mx,label])=>{
    const val=_cfg[key]??0;
    const step=typ==="int"?1:(mx<=1?0.01:0.1);
    const d=typ==="int"?val:parseFloat(val).toFixed(2);
    c.insertAdjacentHTML("beforeend",`<div class="krow"><label><span class="kn">${label}</span><span class="kv" id="kv-${key}">${d}</span></label>
      <input type="range" class="ora-thumb" min="${mn}" max="${mx}" step="${step}" value="${val}" id="kr-${key}" oninput="onKnob('${key}','${typ}',this.value)"></div>`);
  });
}
function onKnob(key,typ,raw){
  const v=typ==="int"?parseInt(raw):parseFloat(raw);_cfg[key]=v;
  const el=document.getElementById("kv-"+key);if(el)el.textContent=typ==="int"?v:v.toFixed(2);
}
function collectGroup(group){
  const p={};
  SCHEMA.filter(s=>s[5]===group).forEach(([key,typ])=>{
    const el=document.getElementById("kr-"+key);if(!el)return;
    p[key]=typ==="int"?parseInt(el.value):parseFloat(el.value);
  });
  return p;
}
async function applyGroup(group){
  const id="apply"+group.charAt(0).toUpperCase()+group.slice(1);
  const btn=document.getElementById(id),fb=document.getElementById("fb-"+group);
  if(btn){btn.classList.add("busy");btn.textContent="Sending…";}
  try{
    const r=await fetch("/api/full_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collectGroup(group))});
    const d=await r.json();
    showFb(fb,d.ok?"ok":"err",d.ok?"✓ Applied":"Server error");
  }catch{showFb(fb,"err","Request failed");}
  finally{if(btn){btn.classList.remove("busy");btn.textContent="▶ Apply";}}
}

/* ── ADVANCED ── */
async function applyAdvanced(){
  const btn=document.getElementById("applyAdvanced"),fb=document.getElementById("fb-advanced");
  btn.classList.add("busy");btn.textContent="Sending…";
  const fc={...collectGroup("fusion"),...collectGroup("shape"),...collectGroup("zoom")};
  const cv={
    contour_min_circularity:parseFloat(document.getElementById("sl-circ").value),
    max_aspect_ratio:parseFloat(document.getElementById("sl-asr").value),
    watershed_fg_thresh:parseFloat(document.getElementById("sl-wfg").value),
    nms_iou_threshold:parseFloat(document.getElementById("sl-nms").value),
  };
  try{
    const[r1,r2]=await Promise.all([
      fetch("/api/full_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(fc)}),
      fetch("/api/cv_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cv)}),
    ]);
    const[d1,d2]=await Promise.all([r1.json(),r2.json()]);
    showFb(fb,(d1.ok&&d2.ok)?"ok":"err",(d1.ok&&d2.ok)?"✓ Applied":"Partial error");
  }catch{showFb(fb,"err","Request failed");}
  finally{btn.classList.remove("busy");btn.textContent="▶ Apply all";}
}
function resetAdvanced(){
  ["fusion","shape","zoom"].forEach(g=>SCHEMA.filter(s=>s[5]===g).forEach(([key,typ,mn,mx,lbl,grp,def])=>{
    _cfg[key]=def;
    const s=document.getElementById("kr-"+key),v=document.getElementById("kv-"+key);
    if(s)s.value=def;if(v)v.textContent=typ==="int"?def:parseFloat(def).toFixed(2);
  }));
  const sv=(id,val,fmt)=>{const e=document.getElementById(id);if(e)e.value=val;const v=document.getElementById("v-"+id.slice(3));if(v)v.textContent=fmt(val);};
  sv("sl-circ",SHP_DEF.contour_min_circularity,x=>x.toFixed(2));
  sv("sl-asr",SHP_DEF.max_aspect_ratio,x=>x.toFixed(1));
  sv("sl-wfg",SHP_DEF.watershed_fg_thresh,x=>x.toFixed(2));
  sv("sl-nms",SHP_DEF.nms_iou_threshold,x=>x.toFixed(2));
  applyAdvanced();
}
function resetGroup(group){
  SCHEMA.filter(s=>s[5]===group).forEach(([key,typ,mn,mx,lbl,grp,def])=>{
    _cfg[key]=def;
    const s=document.getElementById("kr-"+key),v=document.getElementById("kv-"+key);
    if(s)s.value=def;if(v)v.textContent=typ==="int"?def:parseFloat(def).toFixed(2);
  });
  applyGroup(group);
}

/* ── AUTO-MOVE TOGGLE ── */
let _autoMoveEnabled = true;

async function onAutoMoveToggle(enabled) {
  _autoMoveEnabled = enabled;
  try {
    await fetch("/api/auto_move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: enabled })
    });
    addLine(`Auto-move ${enabled ? "enabled" : "disabled"} (arm + gripper)`);
  } catch(e) { console.error(e); }
}

async function loadAutoMoveState() {
  try {
    const r = await fetch("/api/auto_move");
    const d = await r.json();
    _autoMoveEnabled = d.enabled;
    document.getElementById("autoMoveToggle").checked = _autoMoveEnabled;
  } catch(e) { console.warn("Failed to load auto-move state:", e); }
}

/* ── INIT ── */
async function initUI(){
  try{
    const[cr,ar,cvr,fr]=await Promise.all([
      fetch("/api/cv_config"),fetch("/api/ai_enabled"),fetch("/api/cv_enabled"),fetch("/api/full_config")
    ]);
    const[cfg,ai,cv,full]=await Promise.all([cr.json(),ar.json(),cvr.json(),fr.json()]);
    loadCVConfig(cfg);setAiUI(ai.enabled);setCvUI(cv.enabled);updateComboBtn();
    Object.assign(_cfg,full);
    ["thresholds","fusion","tracking","shape","zoom"].forEach(buildKnobs);
    await loadAutoMoveState();
  }catch(e){console.warn("initUI failed:",e);}
}
function loadCVConfig(cfg){
  const map={"sl-h1l":cfg.h1_low,"sl-h1h":cfg.h1_high,"sl-h2l":cfg.h2_low,"sl-h2h":cfg.h2_high,"sl-sat":cfg.sat_min,"sl-vmin":cfg.val_min,"sl-vmax":cfg.val_max};
  for(const[id,val]of Object.entries(map)){const e=document.getElementById(id);if(e)e.value=val;}
  if(cfg.contour_min_circularity!=null){document.getElementById("sl-circ").value=cfg.contour_min_circularity;document.getElementById("v-circ").textContent=parseFloat(cfg.contour_min_circularity).toFixed(2);}
  if(cfg.max_aspect_ratio!=null){document.getElementById("sl-asr").value=cfg.max_aspect_ratio;document.getElementById("v-asr").textContent=parseFloat(cfg.max_aspect_ratio).toFixed(1);}
  if(cfg.watershed_fg_thresh!=null){document.getElementById("sl-wfg").value=cfg.watershed_fg_thresh;document.getElementById("v-wfg").textContent=parseFloat(cfg.watershed_fg_thresh).toFixed(2);}
  if(cfg.nms_iou_threshold!=null){document.getElementById("sl-nms").value=cfg.nms_iou_threshold;document.getElementById("v-nms").textContent=parseFloat(cfg.nms_iou_threshold).toFixed(2);}
  updateHSVLabels();
}

/* ── AI TOGGLE ── */
function setAiUI(on){_aiOn=on;document.getElementById("aiToggle").checked=on;document.getElementById("aiSub").textContent=on?"Active — fusing with CV":"Disabled — CV only";updateComboBtn();}
async function onAiToggle(on){setAiUI(on);try{await fetch("/api/ai_enabled",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:on})});}catch(e){console.error(e);}}

/* ── CV TOGGLE ── */
function setCvUI(on){_cvOn=on;document.getElementById("cvToggle").checked=on;document.getElementById("cvSub").textContent=on?"Active — colour detection":"Disabled — AI only";updateComboBtn();}
async function onCvToggle(on){setCvUI(on);try{await fetch("/api/cv_enabled",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:on})});}catch(e){console.error(e);}}

/* ── COMBO TOGGLE ── */
function updateComboBtn(){
  const btn=document.getElementById("comboBtn"),state=document.getElementById("comboState"),sub=document.getElementById("comboSub");
  btn.className="combo-btn";
  if(_aiOn&&_cvOn){btn.classList.add("combo-both-on");state.textContent="BOTH ON";sub.textContent="Click to disable both";}
  else if(_aiOn||_cvOn){btn.classList.add("combo-mixed");state.textContent=_aiOn?"AI only":"CV only";sub.textContent="Mixed — click to disable both";}
  else{btn.classList.add("combo-both-off");state.textContent="BOTH OFF";sub.textContent="Click to enable both";}
}
async function onComboToggle(){
  const t=!(_aiOn||_cvOn);
  try{
    await Promise.all([
      fetch("/api/ai_enabled",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:t})}),
      fetch("/api/cv_enabled",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:t})}),
    ]);
    setAiUI(t);setCvUI(t);
  }catch(e){console.error(e);}
}

/* ── PRESETS ── */
async function applyPreset(name){
  document.querySelectorAll(".pbtn").forEach(b=>b.classList.remove("on"));
  document.querySelector(`[data-preset="${name}"]`)?.classList.add("on");
  try{
    const r=await fetch("/api/cv_preset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({preset:name})});
    const d=await r.json();
    if(d.ok){loadCVConfig(d.config);showFb(document.getElementById("fb-hsv"),"ok",`✓ ${name}`);}
    else showFb(document.getElementById("fb-hsv"),"err",d.error||"Error");
  }catch{showFb(document.getElementById("fb-hsv"),"err","Request failed");}
}

/* ── HSV ── */
function sliderChanged(){updateHSVLabels();document.querySelectorAll(".pbtn").forEach(b=>b.classList.remove("on"));}
function updateHSVLabels(){
  [["sl-h1l","v-h1l","hm1l"],["sl-h1h","v-h1h","hm1h"],["sl-h2l","v-h2l","hm2l"],["sl-h2h","v-h2h","hm2h"]].forEach(([s,v,m])=>{
    const val=+document.getElementById(s).value;
    document.getElementById(v).textContent=val;
    const mk=document.getElementById(m);if(mk)mk.style.left=(val/179*100)+"%";
  });
  document.getElementById("v-sat").textContent=document.getElementById("sl-sat").value;
  document.getElementById("v-vmin").textContent=document.getElementById("sl-vmin").value;
  document.getElementById("v-vmax").textContent=document.getElementById("sl-vmax").value;
}
async function applyHSV(){
  const btn=document.getElementById("applyHSVBtn"),fb=document.getElementById("fb-hsv");
  btn.classList.add("busy");btn.textContent="Sending…";
  const p={h1_low:+document.getElementById("sl-h1l").value,h1_high:+document.getElementById("sl-h1h").value,
            h2_low:+document.getElementById("sl-h2l").value,h2_high:+document.getElementById("sl-h2h").value,
            sat_min:+document.getElementById("sl-sat").value,val_min:+document.getElementById("sl-vmin").value,val_max:+document.getElementById("sl-vmax").value};
  try{
    const r=await fetch("/api/cv_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
    const d=await r.json();showFb(fb,d.ok?"ok":"err",d.ok?"✓ Applied":"Server error");
  }catch{showFb(fb,"err","Request failed");}
  finally{btn.classList.remove("busy");btn.textContent="▶ Apply";}
}
function resetHSV(){
  const m={"sl-h1l":HSV_DEF.h1_low,"sl-h1h":HSV_DEF.h1_high,"sl-h2l":HSV_DEF.h2_low,"sl-h2h":HSV_DEF.h2_high,"sl-sat":HSV_DEF.sat_min,"sl-vmin":HSV_DEF.val_min,"sl-vmax":HSV_DEF.val_max};
  for(const[id,v]of Object.entries(m)){const e=document.getElementById(id);if(e)e.value=v;}
  updateHSVLabels();
  document.querySelectorAll(".pbtn").forEach(b=>b.classList.remove("on"));
  document.querySelector('[data-preset="red"]')?.classList.add("on");
  applyHSV();
}
function showFb(el,cls,msg){if(!el)return;el.className="fb "+cls;el.textContent=msg;setTimeout(()=>{el.className="fb";el.textContent="";},3000);}

/* ── SENSORS: dead-reckoning bar helper ── */
function _drBar(estDeg, minDeg, maxDeg) {
  // Draw a centred bar: left half = negative travel, right half = positive
  // If no limits, use ±90° as display range
  const lo = minDeg ?? -90, hi = maxDeg ?? 90;
  const range = Math.max(hi - lo, 1);
  const zero_pct = (-lo / range) * 100;          // where 0° sits on the bar
  const cur_pct  = ((estDeg - lo) / range) * 100; // where current pos sits
  // bar goes from zero to current (might go left or right)
  const left  = Math.min(zero_pct, cur_pct);
  const width = Math.abs(cur_pct - zero_pct);
  return `<div class="sarc-dr">
    <div class="sarc-dr-f" style="left:${left.toFixed(1)}%;width:${width.toFixed(1)}%"></div>
  </div>`;
}

/* ── SENSORS: build one card (sensor + optional DR section) ── */
const SLBL={turntable:"Turntable (ID 13)",lift:"Lift (ID 3+4, dual)",gripper:"Gripper (ID 8)",arm:"Arm (ID 5)",pivot:"Pivot (ID 2)"};

function _scard(name, sensorInfo, drInfo) {
  const lbl   = SLBL[name] || name;
  const alive = sensorInfo.available;

  // ── sensor portion ──
  let rows = "";
  if (alive && sensorInfo.data) {
    const d = sensorInfo.data, pct = ((d.deg / 360) * 100).toFixed(0);
    rows = `<div class="srow"><span class="sk">Angle</span><span class="sv">${parseFloat(d.deg).toFixed(1)}°</span></div>
            <div class="srow"><span class="sk">Raw</span><span class="sv">${d.raw}</span></div>
            <div class="srow"><span class="sk">Laps</span><span class="sv">${d.laps >= 0 ? "+" : ""}${d.laps}</span></div>
            <div class="sarc"><div class="sarc-f" style="width:${pct}%"></div></div>`;
  } else if (alive && sensorInfo.channels) {
    for (const [ch, d] of Object.entries(sensorInfo.channels)) {
      if (!d) { rows += `<div class="srow"><span class="sk" style="color:var(--blu)">Ch${ch}</span><span class="sv" style="color:var(--mut)">—</span></div>`; continue; }
      const pct = ((d.deg / 360) * 100).toFixed(0);
      rows += `<div class="srow" style="margin-top:5px"><span class="sk" style="color:var(--blu);font-weight:600">Ch${ch}</span>
               <span class="sv">${parseFloat(d.deg).toFixed(1)}° raw ${d.raw} laps ${d.laps >= 0 ? "+" : ""}${d.laps}</span></div>
               <div class="sarc"><div class="sarc-f" style="width:${pct}%"></div></div>`;
    }
  } else if (sensorInfo.error) {
    rows = `<div class="srow"><span class="sk" style="color:var(--acc)">${sensorInfo.error}</span></div>`;
  }

  // ── dead-reckoning portion (only for motors that have DR data) ──
  let drHtml = "";
  if (drInfo && !drInfo.error) {
    const estDeg  = drInfo.estimated_deg;
    const rawAcc  = drInfo.dead_pos;
    const zeroDeg = drInfo.zero_deg;
    const minDeg  = drInfo.min_deg;
    const maxDeg  = drInfo.max_deg;
    const s2d     = drInfo.speed_to_deg;

    // Only show DR section when sensor is absent (it's the primary position source then)
    // but always show it so the user can see accumulator drift
    const estStr  = estDeg !== null ? `${estDeg >= 0 ? "+" : ""}${estDeg.toFixed(1)}°` : "—";
    const rangeStr = (minDeg !== null && maxDeg !== null) ? `${minDeg}° … ${maxDeg}°` : "unlimited";

    let posBar = "";
    if (estDeg !== null) posBar = _drBar(estDeg, minDeg, maxDeg);

    drHtml = `<div class="dr-section">
      <div class="dr-label">Dead-reckoning${drInfo.has_sensor ? " (sensor primary)" : " (open-loop)"}</div>
      <div class="srow"><span class="sk">Est. position</span><span class="sv" style="color:var(--pur)">${estStr}</span></div>
      <div class="srow"><span class="sk">Accumulator</span><span class="sv">${rawAcc.toFixed(3)}</span></div>
      ${zeroDeg !== null ? `<div class="srow"><span class="sk">Zero point</span><span class="sv">${zeroDeg >= 0 ? "+" : ""}${zeroDeg.toFixed(1)}°</span></div>` : ""}
      <div class="srow"><span class="sk">Travel range</span><span class="sv" style="font-size:9px">${rangeStr}</span></div>
      ${s2d !== null ? `<div class="srow"><span class="sk">speed_to_deg</span><span class="sv">${s2d}</span></div>` : ""}
      ${posBar}
    </div>`;
  }

  // card class: sensor live → green border; DR-only → purple border; nothing → dead
  const cardCls = alive ? "live" : (drInfo && !drInfo.error && !alive) ? "dr" : "dead";
  const badgeCls = alive ? "on" : (drInfo && !drInfo.error && !alive) ? "dr" : "";
  const badgeTxt = alive ? "LIVE" : (drInfo && !drInfo.error) ? "DR ONLY" : (sensorInfo.error ? "ERROR" : "NO SENSOR");

  return `<div class="scard ${cardCls}">
    <div class="scard-h">
      <span class="sname">${lbl}</span>
      <span class="sbadge ${badgeCls}">${badgeTxt}</span>
    </div>
    ${rows}${drHtml}
  </div>`;
}

async function refreshSensors() {
  const c = document.getElementById("sensorList"); if (!c) return;
  try {
    const [sr, dr] = await Promise.all([
      fetch("/api/corner_sensors"),
      fetch("/api/dead_reckoning"),
    ]);
    const [sensors, deadReckon] = await Promise.all([sr.json(), dr.json()]);

    // Merge: all motors that appear in either response
    const allNames = new Set([...Object.keys(sensors), ...Object.keys(deadReckon)]);
    let html = "";
    for (const name of allNames) {
      const sInfo = sensors[name]     || { available: false };
      const dInfo = deadReckon[name]  || null;
      html += _scard(name, sInfo, dInfo);
    }
    c.innerHTML = html || `<div style="font-size:10px;color:var(--mut);padding:5px 0">No sensors found.</div>`;
  } catch(e) {
    c.innerHTML = `<div style="color:var(--acc);font-size:10px;padding:5px 0">Request failed: ${e}</div>`;
  }
}

const STATUS_COLORS={
  "STOP":"var(--mut)","FORWARD":"var(--grn)","BACKWARD":"var(--blu)",
  "LEFT":"var(--grn)","RIGHT":"var(--blu)","UP":"var(--grn)","DOWN":"var(--blu)",
  "GRIP":"var(--blu)","GRIPPED":"var(--blu)","OPEN":"var(--yel)",
  "BUSY":"var(--ora)","EXTENDING":"var(--grn)",
};
function _servoCard(s){
  const sc=STATUS_COLORS[s.status]||"var(--txt)";
  const hw=s.simulated?`<span style="color:var(--ora)">SIM</span>`:`<span style="color:var(--grn)">REAL</span>`;
  const speed=s.speed>0?s.speed:"—";
  const alive=s.status!=="STOP"||!s.simulated;
  return`<div class="scard ${alive?"live":""}"><div class="scard-h"><span class="sname">ID${String(s.id).padStart(2,"0")} – ${s.name}</span><span class="sbadge" style="color:${sc};border-color:${sc}">${s.status}</span></div><div class="srow"><span class="sk">Speed</span><span class="sv">${speed}</span></div><div class="srow"><span class="sk">Hardware</span><span class="sv">${hw}</span></div></div>`;
}
async function refreshServos(){
  const c=document.getElementById("servoList");if(!c)return;
  try{
    const r=await fetch("/api/servo_status");
    const d=await r.json();
    c.innerHTML=d.length?d.map(_servoCard).join(""):`<div style="font-size:10px;color:var(--mut);padding:5px 0">No servos reported.</div>`;
  }catch(e){c.innerHTML=`<div style="color:var(--acc);font-size:10px">Request failed: ${e}</div>`;}
}

/* ── DRAG RESIZE ── */
function makeDrag(divId,panelId,side){
  const div=document.getElementById(divId),panel=document.getElementById(panelId);
  if(!div||!panel)return;
  div.addEventListener("mousedown",e=>{
    e.preventDefault();div.classList.add("drag");
    document.body.style.cursor="col-resize";document.body.style.userSelect="none";
    const main=document.querySelector("main").getBoundingClientRect();
    const onMove=ev=>{let w=side==="left"?Math.max(200,Math.min(window.innerWidth*.45,ev.clientX-main.left)):Math.max(160,Math.min(window.innerWidth*.55,main.right-ev.clientX));panel.style.width=w+"px";};
    const onUp=()=>{div.classList.remove("drag");document.body.style.cursor="";document.body.style.userSelect="";window.removeEventListener("mousemove",onMove);window.removeEventListener("mouseup",onUp);};
    window.addEventListener("mousemove",onMove);window.addEventListener("mouseup",onUp);
  });
}
makeDrag("divLeft","settingsPanel","left");
makeDrag("divRight","consolePanel","right");

/* ── STATS ── */
const RE_FPS=/FPS:\s*(\d+).*?AI:\s*(\d+).*?CV:\s*(\d+).*?Fused:\s*(\d+).*?Hits:\s*(\d+).*?Possible:\s*(\d+)/;
const RE_CAM=/Camera mode:\s*(\w+)/;
function parseStats(line){
  let m=RE_FPS.exec(line);
  if(m)[["s-fps",m[1]],["s-ai",m[2]],["s-cv",m[3]],["s-fused",m[4]],["s-hits",m[5]],["s-poss",m[6]]].forEach(([id,v])=>{document.getElementById(id).textContent=v;});
  m=RE_CAM.exec(line);if(m){document.getElementById("s-cam").textContent=m[1];document.getElementById("camLabel").textContent="CAM: "+m[1].toUpperCase();}
}

/* ── CONSOLE ── */
function cls(l){if(/FPS:/i.test(l))return"fps";if(/verbonden!|connected/i.test(l))return"ok";if(/error|failed/i.test(l))return"err";if(/warn/i.test(l))return"warn";if(/Dashboard:|http:\/\//i.test(l))return"info";if(/^──/.test(l.trim()))return"sep";return"";}
const logEl=document.getElementById("log");
function addLine(text){
  if(text===lastText&&lastMsgEl){lastCount++;let b=lastMsgEl.querySelector(".rbadge");if(!b){b=document.createElement("span");b.className="rbadge";lastMsgEl.appendChild(b);}b.textContent="×"+lastCount;if(autoScroll)logEl.scrollTop=logEl.scrollHeight;return;}
  while(logEl.children.length>=MAX_LINES)logEl.removeChild(logEl.firstChild);
  const ts=new Date().toTimeString().slice(0,8),row=document.createElement("div");
  row.className="ll";
  const sp=document.createElement("span");sp.className="lm "+cls(text);sp.textContent=text;
  row.innerHTML=`<span class="lt">${ts}</span>`;row.appendChild(sp);logEl.appendChild(row);
  lastText=text;lastCount=1;lastMsgEl=sp;lineCount++;
  document.getElementById("lc").textContent=lineCount+" regels";
  if(autoScroll)logEl.scrollTop=logEl.scrollHeight;
}
function clearLog(){logEl.innerHTML="";lineCount=0;lastText=null;lastMsgEl=null;document.getElementById("lc").textContent="0 regels";}
function toggleScroll(){autoScroll=!autoScroll;const b=document.getElementById("scrollBtn");b.textContent=autoScroll?"↓ Auto":"⏸ Gepauzeerd";b.className=autoScroll?"on":"";if(autoScroll)logEl.scrollTop=logEl.scrollHeight;}
logEl.addEventListener("scroll",()=>{if(logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight<40&&!autoScroll)return;if(logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight>=40&&autoScroll){autoScroll=false;document.getElementById("scrollBtn").textContent="⏸ Gepauzeerd";document.getElementById("scrollBtn").className="";}});

/* ── LIVE DOT ── */
function setLive(on,text){document.getElementById("liveDot").className=on?"dot on":"dot";document.getElementById("connDot").className=on?"dot on":"dot";document.getElementById("liveText").textContent=on?"LIVE":text;document.getElementById("connText").textContent=text;}

/* ── CLOCK ── */
setInterval(()=>{document.getElementById("timeLabel").textContent=new Date().toLocaleTimeString("nl-NL");},1000);

/* ── VIDEO ── */
const feed=document.getElementById("feed"),noSignal=document.getElementById("noSignal");
let feedAlive=false,feedRetry=null,feedDog=null,feedDelay=3000;
function _clearFT(){clearTimeout(feedRetry);clearTimeout(feedDog);feedRetry=feedDog=null;}
function _armDog(){clearTimeout(feedDog);feedDog=setTimeout(()=>{addLine("── feed watchdog: herverbinden ──");loadFeed();},12000);}
function loadFeed(){_clearFT();requestAnimationFrame(()=>{feed.src="/video_feed?"+Date.now();feedRetry=setTimeout(()=>{if(!feedAlive)onFeedErr();},10000);});}
function onFeedLoad(){_clearFT();feedAlive=true;feedDelay=3000;feed.style.display="block";feed.style.opacity="1";noSignal.style.display="none";_armDog();}
function onFeedErr(){_clearFT();feedAlive=false;feed.style.display="block";feed.style.opacity="0.18";noSignal.style.display="flex";feedRetry=setTimeout(loadFeed,feedDelay);feedDelay=Math.min(feedDelay*1.5,15000);}
feed.addEventListener("load",onFeedLoad);feed.addEventListener("error",onFeedErr);
document.addEventListener("visibilitychange",()=>{if(!document.hidden)loadFeed();});
window.addEventListener("focus",()=>{if(feedAlive)loadFeed();});
loadFeed();

/* ── SSE ── */
function connect(){
  if(evtSrc){evtSrc.close();evtSrc=null;}
  evtSrc=new EventSource("/logs");
  evtSrc.onopen=()=>{setLive(true,"Verbonden met log stream");addLine("── log stream verbonden ──");};
  evtSrc.onmessage=e=>{parseStats(e.data);addLine(e.data);};
  evtSrc.onerror=()=>{setLive(false,`Verbroken – opnieuw over ${RECONNECT_MS/1000}s…`);evtSrc.close();evtSrc=null;setTimeout(connect,RECONNECT_MS);};
}
connect();
initUI();
updateHSVLabels();

let _killActive=false;
async function triggerHome(){
  const btn=document.getElementById("homeBtn");btn.textContent="⏳ Homing…";btn.disabled=true;
  try{await fetch("/api/home",{method:"POST"});addLine("── home_all() triggered ──");setTimeout(()=>{btn.textContent="🏠 Home";btn.disabled=false;},8000);}
  catch(e){btn.textContent="🏠 Home";btn.disabled=false;addLine("ERROR: home request failed");}
}
async function toggleKill(){
  _killActive=!_killActive;const btn=document.getElementById("killBtn");
  try{
    await fetch("/api/kill",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enabled:_killActive})});
    btn.textContent=_killActive?"💀 KILLED":"☠ KILL";
    btn.style.background=_killActive?"rgba(255,61,90,.3)":"rgba(255,61,90,.08)";
    addLine(_killActive?"── KILL SWITCH ACTIVE ──":"── kill switch released ──");
  }catch(e){_killActive=!_killActive;addLine("ERROR: kill request failed");}
}
</script>
</body>
</html>
"""
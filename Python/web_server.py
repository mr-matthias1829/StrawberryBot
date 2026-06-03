"""Lightweight web dashboard voor de Strawberry Fusion Detector.

Publieke API
-----------
start(host, port)              – start Flask in een daemon-thread (eenmalig)
push_frame(bgr)                – voed een geannoteerd BGR numpy-array aan de MJPEG-stream
push_log(line)                 – stuur een logregel naar alle SSE-clients

REST API (JSON)
---------------
GET  /api/cv_config            – huidige CVConfig als JSON
POST /api/cv_config            – update CVConfig velden (partial update ok)
GET  /api/ai_enabled           – {"enabled": true/false}
POST /api/ai_enabled           – {"enabled": true/false}
POST /api/cv_preset            – {"preset": "red"|"green"|"yellow"|"blue"|"orange"|"custom"}
GET  /api/full_config          – alle config.py knobs als JSON
POST /api/full_config          – partial update van config.py knobs (setattr live)
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

_frame_lock:   threading.Lock    = threading.Lock()
_latest_jpeg:  bytes | None      = None

_subs_lock:    threading.Lock    = threading.Lock()
_subs:         List[queue.Queue] = []

_started    = False
_start_lock = threading.Lock()


# =============================================================================
# COLOUR PRESETS
# =============================================================================

_PRESETS = {
    "red":    dict(h1_low=0,   h1_high=10,  h2_low=160, h2_high=179, sat_min=80,  val_min=50,  val_max=240),
    "orange": dict(h1_low=8,   h1_high=25,  h2_low=0,   h2_high=0,   sat_min=100, val_min=60,  val_max=240),
    "yellow": dict(h1_low=22,  h1_high=38,  h2_low=0,   h2_high=0,   sat_min=80,  val_min=60,  val_max=240),
    "green":  dict(h1_low=35,  h1_high=85,  h2_low=0,   h2_high=0,   sat_min=60,  val_min=40,  val_max=240),
    "blue":   dict(h1_low=100, h1_high=130, h2_low=0,   h2_high=0,   sat_min=60,  val_min=40,  val_max=240),
    "purple": dict(h1_low=125, h1_high=155, h2_low=0,   h2_high=0,   sat_min=60,  val_min=40,  val_max=240),
}

# =============================================================================
# FULL CONFIG — keys exposed to the web UI
# =============================================================================
# Each entry: (key, type, min, max, label, group)
# group: "thresholds" | "fusion" | "tracking" | "shape" | "zoom"

_FULL_CONFIG_SCHEMA = [
    # ── Thresholds ──────────────────────────────────────────────────────────
    ("YOLO_BASE_THRESHOLD",         float, 0.0,  1.0,  "YOLO base threshold",          "thresholds"),
    ("CV_BASE_THRESHOLD",           float, 0.0,  1.0,  "CV base threshold",            "thresholds"),
    ("CV_DIRECT_ACCEPT_THRESHOLD",  float, 0.0,  1.0,  "CV direct-accept threshold",   "thresholds"),
    ("HIGH_AI_CONFIDENCE",          float, 0.0,  1.0,  "High AI confidence",           "thresholds"),
    ("LOW_AI_CONFIDENCE",           float, 0.0,  1.0,  "Low AI confidence",            "thresholds"),
    # ── Fusion weights ───────────────────────────────────────────────────────
    ("YOLO_FUSION_WEIGHT",          float, 0.0,  1.0,  "YOLO fusion weight",           "fusion"),
    ("CV_FUSION_WEIGHT",            float, 0.0,  1.0,  "CV fusion weight",             "fusion"),
    # ── CV scoring weights ───────────────────────────────────────────────────
    ("CV_WEIGHT_REDNESS",           float, 0.0,  1.0,  "CV weight: redness",           "fusion"),
    ("CV_WEIGHT_CIRCULARITY",       float, 0.0,  1.0,  "CV weight: circularity",       "fusion"),
    ("CV_WEIGHT_SIZE",              float, 0.0,  1.0,  "CV weight: size",              "fusion"),
    ("CV_WEIGHT_TEXTURE",           float, 0.0,  1.0,  "CV weight: texture",           "fusion"),
    # ── Tracking / persistence ───────────────────────────────────────────────
    ("PERSISTENCE_REQUIRED",        int,   1,    20,   "Frames to confirm (AI/fused)", "tracking"),
    ("PERSISTENCE_REQUIRED_CV_ONLY",int,   1,    20,   "Frames to confirm (CV only)",  "tracking"),
    ("PERSISTENCE_DECAY",           float, 0.0,  1.0,  "Confidence decay per miss",    "tracking"),
    ("IOU_MATCH_THRESHOLD",         float, 0.0,  1.0,  "IoU match threshold",          "tracking"),
    # ── Possible-hit lane ────────────────────────────────────────────────────
    ("POSSIBLE_HIT_MIN_CONF",       float, 0.0,  1.0,  "Possible min conf (fused)",    "tracking"),
    ("POSSIBLE_HIT_MIN_SEEN",       int,   1,    10,   "Possible min seen (fused)",    "tracking"),
    ("POSSIBLE_CV_ONLY_MIN_CONF",   float, 0.0,  1.0,  "Possible min conf (CV only)",  "tracking"),
    ("POSSIBLE_CV_ONLY_MIN_SEEN",   int,   1,    10,   "Possible min seen (CV only)",  "tracking"),
    ("POSSIBLE_AI_ONLY_MIN_CONF",   float, 0.0,  1.0,  "Possible min conf (AI only)",  "tracking"),
    ("POSSIBLE_AI_ONLY_MIN_SEEN",   int,   1,    10,   "Possible min seen (AI only)",  "tracking"),
    ("POSSIBLE_AI_CONF_WEIGHT",     float, 0.0,  1.0,  "AI conf weight for possible",  "tracking"),
    ("POSSIBLE_TARGET_MIN_CONF",    float, 0.0,  1.0,  "Possible target min conf",     "tracking"),
    # ── Shape / contour ──────────────────────────────────────────────────────
    ("MIN_CONTOUR_AREA",            int,   10,   5000, "Min contour area (px²)",       "shape"),
    ("CONVEXITY_MIN_AREA",          int,   100,  20000,"Watershed split min area",     "shape"),
    # ── Zoom recheck ─────────────────────────────────────────────────────────
    ("MAX_RECHECKS",                int,   0,    10,   "Max zoom rechecks",            "zoom"),
    ("ZOOM_SCALE_FACTOR",           float, 1.0,  4.0,  "Zoom scale factor",            "zoom"),
    ("RECHECK_AI_CONF",             float, 0.0,  1.0,  "Zoom recheck AI threshold",    "zoom"),
    ("RECHECK_CV_CONF",             float, 0.0,  1.0,  "Zoom recheck CV threshold",    "zoom"),
]


def _read_full_config() -> dict:
    import config as _cfg
    out = {}
    for key, typ, *_ in _FULL_CONFIG_SCHEMA:
        val = getattr(_cfg, key, None)
        if val is not None:
            out[key] = typ(val)
    return out


def _write_full_config(data: dict) -> dict:
    import config as _cfg
    changed = {}
    for key, typ, mn, mx, *_ in _FULL_CONFIG_SCHEMA:
        if key not in data:
            continue
        try:
            val = typ(data[key])
            val = max(typ(mn), min(typ(mx), val))
            setattr(_cfg, key, val)
            changed[key] = val
        except (ValueError, TypeError):
            pass
    return changed


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
        daemon=True,
        name="flask-dashboard",
    )
    t.start()
    time.sleep(0.4)
    print(f"Dashboard: http://{_local_ip()}:{port}/")


# =============================================================================
# STDOUT INTERCEPTOR
# =============================================================================

class _Tee(io.TextIOBase):
    def __init__(self, wrapped: io.TextIOBase) -> None:
        self._w   = wrapped
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

    def flush(self) -> None:
        self._w.flush()

    def fileno(self) -> int:
        return self._w.fileno()

    def isatty(self) -> bool:
        try:
            return self._w.isatty()
        except Exception:
            return False


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
    "Pragma":        "no-cache",
    "Expires":       "0",
    "X-Accel-Buffering": "no",
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

    return Response(
        sse_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── CV config API ─────────────────────────────────────────────────────────────

@_app.route("/api/cv_config", methods=["GET"])
def api_cv_config_get():
    from detection import get_cv_config
    return jsonify(get_cv_config().to_dict())


@_app.route("/api/cv_config", methods=["POST"])
def api_cv_config_post():
    from detection import get_cv_config, set_cv_config, CVConfig
    data = request.get_json(force=True, silent=True) or {}
    current = get_cv_config()
    merged  = {**current.to_dict(), **data}
    new_cfg = CVConfig.from_dict(merged)
    set_cv_config(new_cfg)
    print(f"[WebUI] CV config updated: {new_cfg.to_dict()}")
    return jsonify({"ok": True, "config": new_cfg.to_dict()})


@_app.route("/api/cv_preset", methods=["POST"])
def api_cv_preset():
    from detection import get_cv_config, set_cv_config, CVConfig
    data   = request.get_json(force=True, silent=True) or {}
    preset = data.get("preset", "").lower()
    if preset not in _PRESETS:
        return jsonify({"ok": False, "error": f"Unknown preset '{preset}'. Valid: {list(_PRESETS)}"}), 400
    current = get_cv_config()
    merged  = {**current.to_dict(), **_PRESETS[preset]}
    new_cfg = CVConfig.from_dict(merged)
    set_cv_config(new_cfg)
    print(f"[WebUI] CV preset applied: {preset}")
    return jsonify({"ok": True, "preset": preset, "config": new_cfg.to_dict()})


# ── Full config API ───────────────────────────────────────────────────────────

@_app.route("/api/full_config", methods=["GET"])
def api_full_config_get():
    return jsonify(_read_full_config())


@_app.route("/api/full_config", methods=["POST"])
def api_full_config_post():
    data    = request.get_json(force=True, silent=True) or {}
    changed = _write_full_config(data)
    print(f"[WebUI] Full config updated: {changed}")
    return jsonify({"ok": True, "changed": changed})


# ── AI toggle API ─────────────────────────────────────────────────────────────

@_app.route("/api/ai_enabled", methods=["GET"])
def api_ai_get():
    from fusion_engine import is_ai_enabled
    return jsonify({"enabled": is_ai_enabled()})


@_app.route("/api/ai_enabled", methods=["POST"])
def api_ai_post():
    from fusion_engine import set_ai_enabled
    data    = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    set_ai_enabled(enabled)
    return jsonify({"ok": True, "enabled": enabled})


@_app.route("/favicon.ico")
def route_favicon():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        b'<text y="26" font-size="28">&#x1F353;</text></svg>'
    )
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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strawberry Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #080c10;
      --bg2:       #0d1219;
      --surface:   #111720;
      --border:    #1e2a38;
      --border2:   #243040;
      --accent:    #ff3d5a;
      --green:     #2ddb72;
      --green-lo:  rgba(45,219,114,.12);
      --yellow:    #f5c842;
      --blue:      #3da9f5;
      --orange:    #ff8c42;
      --text:      #c8d6e8;
      --muted:     #5a7080;
      --mono:      'IBM Plex Mono', 'Fira Code', monospace;
      --sans:      'Space Grotesk', system-ui, sans-serif;
      --console-w: 360px;
      --panel-w:   300px;
    }

    html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); }
    body { font-family: var(--mono); font-size: 12px; display: flex; flex-direction: column; }

    /* ─── HEADER ─────────────────────────────────────────── */
    header {
      display: flex; align-items: center; gap: 14px;
      padding: 0 18px; height: 52px;
      background: var(--surface); border-bottom: 1px solid var(--border);
      flex-shrink: 0; overflow: hidden;
    }

    .berry-icon { font-size: 20px; filter: drop-shadow(0 0 6px rgba(255,61,90,.5)); animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-3px)} }

    h1 { font-family: var(--sans); font-size: 14px; font-weight: 700; white-space: nowrap; }
    h1 span { color: var(--accent); }

    .hdivider { width: 1px; height: 26px; background: var(--border); flex-shrink: 0; }

    .live-pill {
      display: flex; align-items: center; gap: 6px;
      padding: 3px 10px; border-radius: 100px;
      border: 1px solid var(--border); background: var(--bg2);
      font-size: 10px; color: var(--muted); white-space: nowrap;
    }
    .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
    .dot.on { background: var(--green); box-shadow: 0 0 7px var(--green); animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    .stats { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; align-items: center; }
    .stat {
      display: flex; flex-direction: column; align-items: center;
      padding: 4px 11px 3px;
      background: var(--bg2); border: 1px solid var(--border); border-radius: 7px;
      min-width: 48px; transition: border-color .3s;
    }
    .stat.flash { border-color: var(--green); }
    .stat-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; line-height: 1; }
    .stat-val  { font-family: var(--sans); font-size: 16px; font-weight: 700; line-height: 1.4; color: var(--text); }
    #s-fps  { color: var(--green); }
    #s-hits { color: var(--accent); }
    #s-cam  { font-size: 11px; font-family: var(--mono); }

    /* ─── MAIN LAYOUT ─────────────────────────────────────── */
    main { flex: 1; display: flex; min-height: 0; }

    /* ─── VIDEO PANEL ─────────────────────────────────────── */
    .video-panel {
      flex: 1; min-width: 0; position: relative;
      background: var(--bg); display: flex; align-items: center; justify-content: center; overflow: hidden;
    }
    .video-panel::after {
      content: ''; position: absolute; inset: 0;
      background: repeating-linear-gradient(to bottom,transparent 0,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);
      pointer-events: none; z-index: 2;
    }
    #feed { max-width: 100%; max-height: 100%; object-fit: contain; display: none; z-index: 1; }
    .no-signal { display: flex; flex-direction: column; align-items: center; gap: 12px; color: var(--muted); z-index: 3; pointer-events: none; }
    .no-signal-icon { font-size: 56px; opacity: .15; animation: breathe 3s ease-in-out infinite; }
    @keyframes breathe { 0%,100%{opacity:.15} 50%{opacity:.25} }
    .no-signal p { font-size: 11px; letter-spacing: .5px; }
    .corner-label { position: absolute; z-index: 3; font-size: 9px; color: rgba(255,255,255,.25); letter-spacing: .5px; text-transform: uppercase; }
    .corner-label.tl { top: 8px; left: 10px; }
    .corner-label.tr { top: 8px; right: 10px; }
    .corner-label.bl { bottom: 8px; left: 10px; }

    /* ─── DRAG DIVIDERS ───────────────────────────────────── */
    .drag-divider {
      width: 5px; flex-shrink: 0; background: var(--border);
      cursor: col-resize; position: relative; transition: background .15s; z-index: 10;
    }
    .drag-divider:hover, .drag-divider.dragging { background: var(--accent); }
    .drag-divider::after {
      content: ''; position: absolute; top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      width: 1px; height: 40px; background: rgba(255,255,255,.1); border-radius: 1px;
    }

    /* ─── SETTINGS PANEL ──────────────────────────────────── */
    .settings-panel {
      width: var(--panel-w); min-width: 220px; max-width: 60vw;
      flex-shrink: 0; display: flex; flex-direction: column;
      background: var(--surface); border-right: 1px solid var(--border); min-height: 0;
    }

    /* ─── TABS ────────────────────────────────────────────── */
    .tab-bar {
      display: flex; flex-shrink: 0;
      border-bottom: 1px solid var(--border);
      overflow-x: auto; scrollbar-width: none;
    }
    .tab-bar::-webkit-scrollbar { display: none; }
    .tab-btn {
      flex-shrink: 0; padding: 0 12px; height: 34px; cursor: pointer;
      font-family: var(--mono); font-size: 9px; color: var(--muted);
      text-transform: uppercase; letter-spacing: .8px;
      background: none; border: none; border-bottom: 2px solid transparent;
      transition: color .15s, border-color .15s;
    }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: var(--text); border-bottom-color: var(--accent); }

    .tab-pane { display: none; flex: 1; overflow-y: auto; }
    .tab-pane.active { display: block; }
    .tab-pane::-webkit-scrollbar { width: 3px; }
    .tab-pane::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

    .section { padding: 12px; border-bottom: 1px solid var(--border); }
    .section-title { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }

    /* AI Toggle */
    .ai-toggle {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 10px; border-radius: 7px;
      border: 1px solid var(--border); background: var(--bg2);
    }
    .ai-label { font-size: 12px; color: var(--text); }
    .ai-label small { display: block; font-size: 9px; color: var(--muted); margin-top: 1px; }

    .toggle-switch { position: relative; width: 36px; height: 20px; flex-shrink: 0; }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .toggle-track {
      position: absolute; inset: 0; border-radius: 10px;
      background: var(--border2); cursor: pointer; transition: background .2s;
    }
    .toggle-track::after {
      content: ''; position: absolute; width: 14px; height: 14px; border-radius: 50%;
      background: var(--muted); top: 3px; left: 3px; transition: all .2s;
    }
    .toggle-switch input:checked + .toggle-track { background: var(--green); }
    .toggle-switch input:checked + .toggle-track::after {
      background: #fff; transform: translateX(16px);
    }

    /* Presets */
    .preset-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; }
    .preset-btn {
      font-family: var(--mono); font-size: 10px; padding: 5px 4px;
      border: 1px solid var(--border); border-radius: 5px;
      background: var(--bg2); color: var(--muted);
      cursor: pointer; text-align: center; transition: all .15s;
    }
    .preset-btn:hover { background: var(--border2); color: var(--text); }
    .preset-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(255,61,90,.08); }
    .preset-btn .pdot { display: block; width: 10px; height: 10px; border-radius: 50%; margin: 0 auto 3px; }

    /* HSV sliders */
    .hsv-row { margin-bottom: 8px; }
    .hsv-row label { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-bottom: 3px; }
    .hsv-row label span { color: var(--text); font-weight: 600; }
    .hsv-row input[type=range] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 4px; border-radius: 2px; outline: none; cursor: pointer;
      background: var(--border2);
    }
    .hsv-row input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 13px; height: 13px; border-radius: 50%;
      background: var(--blue); box-shadow: 0 0 5px rgba(61,169,245,.4); cursor: pointer;
    }
    .hsv-row input[type=range]::-moz-range-thumb {
      width: 13px; height: 13px; border-radius: 50%; border: none;
      background: var(--blue); cursor: pointer;
    }

    /* Hue visual band */
    .hue-band {
      height: 8px; border-radius: 4px; margin-bottom: 10px;
      background: linear-gradient(to right,
        hsl(0,100%,50%), hsl(30,100%,50%), hsl(60,100%,50%),
        hsl(90,100%,50%), hsl(120,100%,50%), hsl(150,100%,50%),
        hsl(180,100%,50%), hsl(210,100%,50%), hsl(240,100%,50%),
        hsl(270,100%,50%), hsl(300,100%,50%), hsl(330,100%,50%), hsl(360,100%,50%));
      position: relative;
    }
    .hue-marker {
      position: absolute; top: -2px; width: 4px; height: 12px;
      background: #fff; border-radius: 2px; transform: translateX(-50%);
      pointer-events: none; transition: left .1s;
    }

    /* Apply button */
    .apply-btn {
      width: 100%; padding: 7px; margin-top: 10px;
      font-family: var(--mono); font-size: 11px; font-weight: 600;
      border: 1px solid var(--green); border-radius: 6px;
      background: var(--green-lo); color: var(--green);
      cursor: pointer; transition: all .15s; letter-spacing: .5px;
    }
    .apply-btn:hover { background: rgba(45,219,114,.25); }
    .apply-btn:active { transform: scale(.97); }
    .apply-btn.busy { opacity: .5; pointer-events: none; }

    .feedback { font-size: 9px; text-align: center; margin-top: 5px; height: 12px; color: var(--muted); }
    .feedback.ok  { color: var(--green); }
    .feedback.err { color: var(--accent); }

    /* ─── KNOB ROWS (full config) ─────────────────────────── */
    .knob-row { margin-bottom: 10px; }
    .knob-row label {
      display: flex; justify-content: space-between; align-items: baseline;
      font-size: 10px; color: var(--muted); margin-bottom: 3px;
    }
    .knob-row label .kname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .knob-row label .kval  { color: var(--text); font-weight: 600; min-width: 36px; text-align: right; }
    .knob-row input[type=range] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 4px; border-radius: 2px; outline: none; cursor: pointer;
      background: var(--border2);
    }
    .knob-row input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 13px; height: 13px; border-radius: 50%;
      background: var(--orange); cursor: pointer;
    }
    .knob-row input[type=range]::-moz-range-thumb {
      width: 13px; height: 13px; border-radius: 50%; border: none;
      background: var(--orange); cursor: pointer;
    }
    .knob-row input[type=number] {
      width: 100%; padding: 3px 6px; background: var(--bg2);
      border: 1px solid var(--border); border-radius: 4px;
      color: var(--text); font-family: var(--mono); font-size: 11px; outline: none;
    }
    .knob-row input[type=number]:focus { border-color: var(--orange); }

    .cfg-apply-btn {
      width: 100%; padding: 7px; margin-top: 4px;
      font-family: var(--mono); font-size: 11px; font-weight: 600;
      border: 1px solid var(--orange); border-radius: 6px;
      background: rgba(255,140,66,.10); color: var(--orange);
      cursor: pointer; transition: all .15s; letter-spacing: .5px;
    }
    .cfg-apply-btn:hover { background: rgba(255,140,66,.22); }
    .cfg-apply-btn:active { transform: scale(.97); }
    .cfg-apply-btn.busy { opacity: .5; pointer-events: none; }

    /* ─── CONSOLE PANEL ───────────────────────────────────── */
    .console-panel {
      width: var(--console-w); min-width: 180px; max-width: 70vw;
      flex-shrink: 0; display: flex; flex-direction: column;
      background: var(--surface); min-height: 0;
    }
    .console-bar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 10px; height: 34px;
      border-bottom: 1px solid var(--border); flex-shrink: 0;
    }
    .console-title { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
    .btns { display: flex; gap: 5px; }

    button {
      font-family: var(--mono); font-size: 10px; padding: 2px 9px;
      border: 1px solid var(--border); border-radius: 4px;
      background: var(--bg2); color: var(--muted); cursor: pointer; transition: all .15s;
    }
    button:hover  { background: var(--border2); color: var(--text); }
    button.active { border-color: var(--green); color: var(--green); background: var(--green-lo); }

    #log { flex: 1; overflow-y: auto; padding: 6px 4px 6px 10px; min-height: 0; }
    #log::-webkit-scrollbar { width: 3px; }
    #log::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

    .ll { display: grid; grid-template-columns: 52px 1fr; gap: 6px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
    .lt { font-size: 9px; color: #2a3a48; padding-top: 3px; }
    .lm { color: var(--text); }
    .lm.fps  { color: var(--green); }
    .lm.err  { color: #ff5370; }
    .lm.warn { color: var(--yellow); }
    .lm.info { color: var(--blue); }
    .lm.ok   { color: var(--green); font-weight: 600; }
    .lm.sep  { color: var(--border2); }

    .repeat-badge {
      display: inline-block; margin-left: 7px;
      font-size: 9px; padding: 0 5px; border-radius: 8px;
      background: var(--border2); color: var(--muted);
      vertical-align: middle; line-height: 16px;
    }

    /* ─── FOOTER ──────────────────────────────────────────── */
    footer {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 14px; height: 28px;
      background: var(--surface); border-top: 1px solid var(--border);
      font-size: 10px; color: var(--muted); flex-shrink: 0;
    }
    #fc { display: flex; align-items: center; gap: 6px; }

    @media (max-width: 960px) {
      main { flex-direction: column; }
      .video-panel { flex: none; height: 40vh; }
      .drag-divider { display: none; }
      .settings-panel { width: 100% !important; min-width: unset; max-height: 280px; border-right: none; border-bottom: 1px solid var(--border); }
      .console-panel  { width: 100% !important; min-width: unset; }
      .stats { display: none; }
    }
  </style>
</head>
<body>

<header>
  <div class="berry-icon">🍓</div>
  <h1>Strawberry <span>Detector</span></h1>
  <div class="hdivider"></div>
  <div class="live-pill">
    <div class="dot" id="liveDot"></div>
    <span id="liveText">Verbinden…</span>
  </div>
  <div class="stats">
    <div class="stat"><span class="stat-lbl">FPS</span><span class="stat-val" id="s-fps">—</span></div>
    <div class="stat"><span class="stat-lbl">AI</span><span class="stat-val" id="s-ai">—</span></div>
    <div class="stat"><span class="stat-lbl">CV</span><span class="stat-val" id="s-cv">—</span></div>
    <div class="stat"><span class="stat-lbl">Fused</span><span class="stat-val" id="s-fused">—</span></div>
    <div class="stat"><span class="stat-lbl">Hits</span><span class="stat-val" id="s-hits">—</span></div>
    <div class="stat"><span class="stat-lbl">Poss.</span><span class="stat-val" id="s-poss">—</span></div>
    <div class="stat"><span class="stat-lbl">Cam</span><span class="stat-val" id="s-cam">—</span></div>
  </div>
</header>

<main>

  <!-- ── SETTINGS PANEL ── -->
  <div class="settings-panel" id="settingsPanel">

    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('mode')">Mode</button>
      <button class="tab-btn" onclick="switchTab('colour')">Colour</button>
      <button class="tab-btn" onclick="switchTab('thresholds')">Thresholds</button>
      <button class="tab-btn" onclick="switchTab('fusion')">Fusion</button>
      <button class="tab-btn" onclick="switchTab('tracking')">Tracking</button>
      <button class="tab-btn" onclick="switchTab('shape')">Shape</button>
      <button class="tab-btn" onclick="switchTab('zoom')">Zoom</button>
    </div>

    <!-- ── TAB: Mode ── -->
    <div class="tab-pane active" id="tab-mode">
      <div class="section">
        <div class="section-title">Detection mode</div>
        <div class="ai-toggle">
          <div class="ai-label">
            AI (YOLO)
            <small id="aiSubLabel">Loading…</small>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" id="aiToggle" onchange="onAiToggle(this.checked)">
            <span class="toggle-track"></span>
          </label>
        </div>
      </div>
    </div>

    <!-- ── TAB: Colour ── -->
    <div class="tab-pane" id="tab-colour">
      <div class="section">
        <div class="section-title">Colour preset</div>
        <div class="preset-grid">
          <button class="preset-btn active" data-preset="red" onclick="applyPreset('red')">
            <span class="pdot" style="background:#e03030"></span>Red
          </button>
          <button class="preset-btn" data-preset="orange" onclick="applyPreset('orange')">
            <span class="pdot" style="background:#e07830"></span>Orange
          </button>
          <button class="preset-btn" data-preset="yellow" onclick="applyPreset('yellow')">
            <span class="pdot" style="background:#d4c030"></span>Yellow
          </button>
          <button class="preset-btn" data-preset="green" onclick="applyPreset('green')">
            <span class="pdot" style="background:#30c050"></span>Green
          </button>
          <button class="preset-btn" data-preset="blue" onclick="applyPreset('blue')">
            <span class="pdot" style="background:#3070e0"></span>Blue
          </button>
          <button class="preset-btn" data-preset="purple" onclick="applyPreset('purple')">
            <span class="pdot" style="background:#9030c0"></span>Purple
          </button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">HSV tuning</div>

        <div style="font-size:9px;color:var(--muted);margin-bottom:4px;">HUE BAND 1 (low-hue / wrap)</div>
        <div class="hue-band">
          <div class="hue-marker" id="hm1l"></div>
          <div class="hue-marker" id="hm1h" style="background:#adf;"></div>
        </div>
        <div class="hsv-row">
          <label>H1 low <span id="v-h1l">0</span></label>
          <input type="range" min="0" max="179" value="0" id="sl-h1l" oninput="sliderChanged()">
        </div>
        <div class="hsv-row">
          <label>H1 high <span id="v-h1h">10</span></label>
          <input type="range" min="0" max="179" value="10" id="sl-h1h" oninput="sliderChanged()">
        </div>

        <div style="font-size:9px;color:var(--muted);margin-bottom:4px;margin-top:6px;">HUE BAND 2 (high-hue / wrap)</div>
        <div class="hue-band">
          <div class="hue-marker" id="hm2l"></div>
          <div class="hue-marker" id="hm2h" style="background:#adf;"></div>
        </div>
        <div class="hsv-row">
          <label>H2 low <span id="v-h2l">160</span></label>
          <input type="range" min="0" max="179" value="160" id="sl-h2l" oninput="sliderChanged()">
        </div>
        <div class="hsv-row">
          <label>H2 high <span id="v-h2h">179</span></label>
          <input type="range" min="0" max="179" value="179" id="sl-h2h" oninput="sliderChanged()">
        </div>

        <div style="font-size:9px;color:var(--muted);margin-bottom:4px;margin-top:6px;">SATURATION / VALUE GATE</div>
        <div class="hsv-row">
          <label>Sat min <span id="v-sat">80</span></label>
          <input type="range" min="0" max="255" value="80" id="sl-sat" oninput="sliderChanged()">
        </div>
        <div class="hsv-row">
          <label>Val min <span id="v-vmin">50</span></label>
          <input type="range" min="0" max="255" value="50" id="sl-vmin" oninput="sliderChanged()">
        </div>
        <div class="hsv-row">
          <label>Val max <span id="v-vmax">240</span></label>
          <input type="range" min="0" max="255" value="240" id="sl-vmax" oninput="sliderChanged()">
        </div>

        <button class="apply-btn" id="applyBtn" onclick="applyHSV()">▶ Apply HSV</button>
        <div class="feedback" id="fb"></div>
      </div>
    </div>

    <!-- ── TAB: Thresholds ── -->
    <div class="tab-pane" id="tab-thresholds">
      <div class="section">
        <div class="section-title">Confidence thresholds</div>
        <div id="knobs-thresholds"></div>
        <button class="cfg-apply-btn" id="applyThresholds" onclick="applyGroup('thresholds')">▶ Apply</button>
        <div class="feedback" id="fb-thresholds"></div>
      </div>
    </div>

    <!-- ── TAB: Fusion ── -->
    <div class="tab-pane" id="tab-fusion">
      <div class="section">
        <div class="section-title">Fusion & CV scoring weights</div>
        <div id="knobs-fusion"></div>
        <button class="cfg-apply-btn" id="applyFusion" onclick="applyGroup('fusion')">▶ Apply</button>
        <div class="feedback" id="fb-fusion"></div>
      </div>
    </div>

    <!-- ── TAB: Tracking ── -->
    <div class="tab-pane" id="tab-tracking">
      <div class="section">
        <div class="section-title">Persistence & possible-hit lane</div>
        <div id="knobs-tracking"></div>
        <button class="cfg-apply-btn" id="applyTracking" onclick="applyGroup('tracking')">▶ Apply</button>
        <div class="feedback" id="fb-tracking"></div>
      </div>
    </div>

    <!-- ── TAB: Shape ── -->
    <div class="tab-pane" id="tab-shape">
      <div class="section">
        <div class="section-title">Contour & shape filters</div>
        <div id="knobs-shape"></div>
        <!-- CVConfig shape params live here too -->
        <div style="margin-top:12px; padding-top:10px; border-top:1px solid var(--border);">
          <div class="section-title" style="margin-bottom:8px;">CVConfig shape params</div>
          <div class="knob-row">
            <label><span class="kname">Circularity min</span><span class="kval" id="v-circ">0.55</span></label>
            <input type="range" min="0" max="1" step="0.01" value="0.55" id="sl-circ" oninput="document.getElementById('v-circ').textContent=parseFloat(this.value).toFixed(2)">
          </div>
          <div class="knob-row">
            <label><span class="kname">Max aspect ratio</span><span class="kval" id="v-asr">1.6</span></label>
            <input type="range" min="1" max="5" step="0.1" value="1.6" id="sl-asr" oninput="document.getElementById('v-asr').textContent=parseFloat(this.value).toFixed(1)">
          </div>
          <div class="knob-row">
            <label><span class="kname">Watershed FG thresh</span><span class="kval" id="v-wfg">0.35</span></label>
            <input type="range" min="0" max="1" step="0.01" value="0.35" id="sl-wfg" oninput="document.getElementById('v-wfg').textContent=parseFloat(this.value).toFixed(2)">
          </div>
          <div class="knob-row">
            <label><span class="kname">NMS IoU threshold</span><span class="kval" id="v-nms">0.35</span></label>
            <input type="range" min="0" max="1" step="0.01" value="0.35" id="sl-nms" oninput="document.getElementById('v-nms').textContent=parseFloat(this.value).toFixed(2)">
          </div>
        </div>
        <button class="cfg-apply-btn" id="applyShape" onclick="applyShape()">▶ Apply</button>
        <div class="feedback" id="fb-shape"></div>
      </div>
    </div>

    <!-- ── TAB: Zoom ── -->
    <div class="tab-pane" id="tab-zoom">
      <div class="section">
        <div class="section-title">Zoom recheck</div>
        <div id="knobs-zoom"></div>
        <button class="cfg-apply-btn" id="applyZoom" onclick="applyGroup('zoom')">▶ Apply</button>
        <div class="feedback" id="fb-zoom"></div>
      </div>
    </div>

  </div><!-- /settings-panel -->

  <div class="drag-divider" id="dragLeft"></div>

  <!-- ── VIDEO PANEL ── -->
  <div class="video-panel">
    <span class="corner-label tl" id="camLabel">CAM</span>
    <span class="corner-label tr" id="resLabel"></span>
    <span class="corner-label bl" id="timeLabel"></span>
    <div class="no-signal" id="noSignal">
      <div class="no-signal-icon">📷</div>
      <p>Wachten op videostream…</p>
    </div>
    <img id="feed" alt="camera">
  </div>

  <div class="drag-divider" id="dragRight"></div>

  <!-- ── CONSOLE PANEL ── -->
  <div class="console-panel" id="consolePanel">
    <div class="console-bar">
      <span class="console-title">Console output</span>
      <div class="btns">
        <button id="scrollBtn" class="active" onclick="toggleScroll()">↓ Auto</button>
        <button onclick="clearLog()">Wis</button>
      </div>
    </div>
    <div id="log"></div>
  </div>

</main>

<footer>
  <div id="fc">
    <div class="dot" id="connDot"></div>
    <span id="connText">Log stream verbinden…</span>
  </div>
  <span id="lc">0 regels</span>
</footer>

<script>
"use strict";

const MAX_LINES    = 600;
const RECONNECT_MS = 3000;

let autoScroll  = true;
let lineCount   = 0;
let evtSrc      = null;
let lastText    = null;
let lastCount   = 1;
let lastMsgEl   = null;
let _dirty      = false;

// Full config schema mirrored from backend
// [key, type, min, max, label, group]
const SCHEMA = [
  ["YOLO_BASE_THRESHOLD",         "float", 0,   1,     "YOLO base threshold",          "thresholds"],
  ["CV_BASE_THRESHOLD",           "float", 0,   1,     "CV base threshold",            "thresholds"],
  ["CV_DIRECT_ACCEPT_THRESHOLD",  "float", 0,   1,     "CV direct-accept threshold",   "thresholds"],
  ["HIGH_AI_CONFIDENCE",          "float", 0,   1,     "High AI confidence",           "thresholds"],
  ["LOW_AI_CONFIDENCE",           "float", 0,   1,     "Low AI confidence",            "thresholds"],
  ["YOLO_FUSION_WEIGHT",          "float", 0,   1,     "YOLO fusion weight",           "fusion"],
  ["CV_FUSION_WEIGHT",            "float", 0,   1,     "CV fusion weight",             "fusion"],
  ["CV_WEIGHT_REDNESS",           "float", 0,   1,     "CV weight: redness",           "fusion"],
  ["CV_WEIGHT_CIRCULARITY",       "float", 0,   1,     "CV weight: circularity",       "fusion"],
  ["CV_WEIGHT_SIZE",              "float", 0,   1,     "CV weight: size",              "fusion"],
  ["CV_WEIGHT_TEXTURE",           "float", 0,   1,     "CV weight: texture",           "fusion"],
  ["PERSISTENCE_REQUIRED",        "int",   1,   20,    "Frames to confirm (AI/fused)", "tracking"],
  ["PERSISTENCE_REQUIRED_CV_ONLY","int",   1,   20,    "Frames to confirm (CV only)",  "tracking"],
  ["PERSISTENCE_DECAY",           "float", 0,   1,     "Confidence decay per miss",    "tracking"],
  ["IOU_MATCH_THRESHOLD",         "float", 0,   1,     "IoU match threshold",          "tracking"],
  ["POSSIBLE_HIT_MIN_CONF",       "float", 0,   1,     "Possible min conf (fused)",    "tracking"],
  ["POSSIBLE_HIT_MIN_SEEN",       "int",   1,   10,    "Possible min seen (fused)",    "tracking"],
  ["POSSIBLE_CV_ONLY_MIN_CONF",   "float", 0,   1,     "Possible min conf (CV only)",  "tracking"],
  ["POSSIBLE_CV_ONLY_MIN_SEEN",   "int",   1,   10,    "Possible min seen (CV only)",  "tracking"],
  ["POSSIBLE_AI_ONLY_MIN_CONF",   "float", 0,   1,     "Possible min conf (AI only)",  "tracking"],
  ["POSSIBLE_AI_ONLY_MIN_SEEN",   "int",   1,   10,    "Possible min seen (AI only)",  "tracking"],
  ["POSSIBLE_AI_CONF_WEIGHT",     "float", 0,   1,     "AI conf weight for possible",  "tracking"],
  ["POSSIBLE_TARGET_MIN_CONF",    "float", 0,   1,     "Possible target min conf",     "tracking"],
  ["MIN_CONTOUR_AREA",            "int",   10,  5000,  "Min contour area (px²)",       "shape"],
  ["CONVEXITY_MIN_AREA",          "int",   100, 20000, "Watershed split min area",     "shape"],
  ["MAX_RECHECKS",                "int",   0,   10,    "Max zoom rechecks",            "zoom"],
  ["ZOOM_SCALE_FACTOR",           "float", 1,   4,     "Zoom scale factor",            "zoom"],
  ["RECHECK_AI_CONF",             "float", 0,   1,     "Zoom recheck AI threshold",    "zoom"],
  ["RECHECK_CV_CONF",             "float", 0,   1,     "Zoom recheck CV threshold",    "zoom"],
];

// Live values cache
const _cfg = {};

// =============================================================================
// TABS
// =============================================================================

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b, i) => {
    const tabs = ["mode","colour","thresholds","fusion","tracking","shape","zoom"];
    b.classList.toggle("active", tabs[i] === name);
  });
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.getElementById("tab-" + name)?.classList.add("active");
}

// =============================================================================
// KNOB BUILDER
// =============================================================================

function buildKnobs(group) {
  const container = document.getElementById("knobs-" + group);
  if (!container) return;
  container.innerHTML = "";
  SCHEMA.filter(s => s[5] === group).forEach(([key, typ, mn, mx, label]) => {
    const val = _cfg[key] ?? 0;
    const step = typ === "int" ? 1 : (mx <= 1 ? 0.01 : 0.1);
    const disp = typ === "int" ? val : parseFloat(val).toFixed(2);
    container.insertAdjacentHTML("beforeend", `
      <div class="knob-row" data-key="${key}" data-type="${typ}" data-min="${mn}" data-max="${mx}">
        <label>
          <span class="kname">${label}</span>
          <span class="kval" id="kv-${key}">${disp}</span>
        </label>
        <input type="range" min="${mn}" max="${mx}" step="${step}" value="${val}"
               id="kr-${key}"
               oninput="onKnob('${key}','${typ}',this.value)">
      </div>
    `);
  });
}

function onKnob(key, typ, raw) {
  const val = typ === "int" ? parseInt(raw) : parseFloat(raw);
  _cfg[key] = val;
  const disp = typ === "int" ? val : val.toFixed(2);
  const el = document.getElementById("kv-" + key);
  if (el) el.textContent = disp;
}

function collectGroup(group) {
  const payload = {};
  SCHEMA.filter(s => s[5] === group).forEach(([key, typ]) => {
    const el = document.getElementById("kr-" + key);
    if (!el) return;
    payload[key] = typ === "int" ? parseInt(el.value) : parseFloat(el.value);
  });
  return payload;
}

async function applyGroup(group) {
  const btn = document.getElementById("apply" + group.charAt(0).toUpperCase() + group.slice(1));
  const fbEl = document.getElementById("fb-" + group);
  if (btn) { btn.classList.add("busy"); btn.textContent = "Sending…"; }
  try {
    const res  = await fetch("/api/full_config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectGroup(group)),
    });
    const data = await res.json();
    if (data.ok) showFeedbackEl(fbEl, "ok", "✓ Applied");
    else showFeedbackEl(fbEl, "err", "Server error");
  } catch(e) {
    showFeedbackEl(fbEl, "err", "Request failed");
  } finally {
    if (btn) { btn.classList.remove("busy"); btn.textContent = "▶ Apply"; }
  }
}

// =============================================================================
// SHAPE TAB (combo: full_config + cv_config shape params)
// =============================================================================

async function applyShape() {
  const btn  = document.getElementById("applyShape");
  const fbEl = document.getElementById("fb-shape");
  btn.classList.add("busy"); btn.textContent = "Sending…";
  try {
    // full_config knobs (MIN_CONTOUR_AREA, CONVEXITY_MIN_AREA)
    const fcPayload = collectGroup("shape");
    // CVConfig shape params
    const cvPayload = {
      contour_min_circularity: parseFloat(document.getElementById("sl-circ").value),
      max_aspect_ratio:        parseFloat(document.getElementById("sl-asr").value),
      watershed_fg_thresh:     parseFloat(document.getElementById("sl-wfg").value),
      nms_iou_threshold:       parseFloat(document.getElementById("sl-nms").value),
    };
    const [r1, r2] = await Promise.all([
      fetch("/api/full_config", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(fcPayload) }),
      fetch("/api/cv_config",   { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(cvPayload) }),
    ]);
    const [d1, d2] = await Promise.all([r1.json(), r2.json()]);
    if (d1.ok && d2.ok) showFeedbackEl(fbEl, "ok", "✓ Applied");
    else showFeedbackEl(fbEl, "err", "Partial error");
  } catch(e) {
    showFeedbackEl(fbEl, "err", "Request failed");
  } finally {
    btn.classList.remove("busy"); btn.textContent = "▶ Apply";
  }
}

// =============================================================================
// INIT — fetch current backend state
// =============================================================================

async function initUI() {
  try {
    const [cfgRes, aiRes, fullRes] = await Promise.all([
      fetch("/api/cv_config"),
      fetch("/api/ai_enabled"),
      fetch("/api/full_config"),
    ]);
    const cfg  = await cfgRes.json();
    const ai   = await aiRes.json();
    const full = await fullRes.json();

    loadConfig(cfg);
    setAiUI(ai.enabled);
    loadFullConfig(full);

  } catch(e) {
    console.warn("Could not fetch initial config:", e);
  }
}

function loadConfig(cfg) {
  const map = {
    "sl-h1l": cfg.h1_low,  "sl-h1h": cfg.h1_high,
    "sl-h2l": cfg.h2_low,  "sl-h2h": cfg.h2_high,
    "sl-sat": cfg.sat_min,
    "sl-vmin": cfg.val_min, "sl-vmax": cfg.val_max,
  };
  for (const [id, val] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }
  // CVConfig shape sliders
  if (cfg.contour_min_circularity !== undefined) {
    document.getElementById("sl-circ").value = cfg.contour_min_circularity;
    document.getElementById("v-circ").textContent = parseFloat(cfg.contour_min_circularity).toFixed(2);
  }
  if (cfg.max_aspect_ratio !== undefined) {
    document.getElementById("sl-asr").value = cfg.max_aspect_ratio;
    document.getElementById("v-asr").textContent = parseFloat(cfg.max_aspect_ratio).toFixed(1);
  }
  if (cfg.watershed_fg_thresh !== undefined) {
    document.getElementById("sl-wfg").value = cfg.watershed_fg_thresh;
    document.getElementById("v-wfg").textContent = parseFloat(cfg.watershed_fg_thresh).toFixed(2);
  }
  if (cfg.nms_iou_threshold !== undefined) {
    document.getElementById("sl-nms").value = cfg.nms_iou_threshold;
    document.getElementById("v-nms").textContent = parseFloat(cfg.nms_iou_threshold).toFixed(2);
  }
  updateLabels();
  _dirty = false;
}

function loadFullConfig(full) {
  Object.assign(_cfg, full);
  // Build all knob groups
  ["thresholds","fusion","tracking","shape","zoom"].forEach(buildKnobs);
}

// =============================================================================
// AI TOGGLE
// =============================================================================

function setAiUI(enabled) {
  const cb  = document.getElementById("aiToggle");
  const lbl = document.getElementById("aiSubLabel");
  cb.checked  = enabled;
  lbl.textContent = enabled ? "Active — fusing with CV" : "Disabled — CV only";
}

async function onAiToggle(enabled) {
  setAiUI(enabled);
  try {
    await fetch("/api/ai_enabled", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({enabled}),
    });
  } catch(e) { console.error("AI toggle failed:", e); }
}

// =============================================================================
// COLOUR PRESETS
// =============================================================================

async function applyPreset(name) {
  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  document.querySelector(`[data-preset="${name}"]`)?.classList.add("active");
  try {
    const res = await fetch("/api/cv_preset", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({preset: name}),
    });
    const data = await res.json();
    if (data.ok) {
      loadConfig(data.config);
      showFeedback("ok", `✓ Preset '${name}' applied`);
    } else {
      showFeedback("err", data.error || "Error");
    }
  } catch(e) {
    showFeedback("err", "Request failed");
  }
}

// =============================================================================
// HSV SLIDERS
// =============================================================================

function sliderChanged() {
  updateLabels();
  _dirty = true;
  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
}

function updateLabels() {
  const fields = [
    ["sl-h1l","v-h1l","hm1l"], ["sl-h1h","v-h1h","hm1h"],
    ["sl-h2l","v-h2l","hm2l"], ["sl-h2h","v-h2h","hm2h"],
  ];
  for (const [sId, vId, mId] of fields) {
    const v = +document.getElementById(sId).value;
    document.getElementById(vId).textContent = v;
    const marker = document.getElementById(mId);
    if (marker) marker.style.left = (v / 179 * 100) + "%";
  }
  document.getElementById("v-sat").textContent  = document.getElementById("sl-sat").value;
  document.getElementById("v-vmin").textContent = document.getElementById("sl-vmin").value;
  document.getElementById("v-vmax").textContent = document.getElementById("sl-vmax").value;
}

async function applyHSV() {
  const btn = document.getElementById("applyBtn");
  btn.classList.add("busy"); btn.textContent = "Sending…";
  const payload = {
    h1_low:  +document.getElementById("sl-h1l").value,
    h1_high: +document.getElementById("sl-h1h").value,
    h2_low:  +document.getElementById("sl-h2l").value,
    h2_high: +document.getElementById("sl-h2h").value,
    sat_min: +document.getElementById("sl-sat").value,
    val_min: +document.getElementById("sl-vmin").value,
    val_max: +document.getElementById("sl-vmax").value,
  };
  try {
    const res  = await fetch("/api/cv_config", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.ok) { showFeedback("ok", "✓ Applied"); _dirty = false; }
    else showFeedback("err", "Server error");
  } catch(e) {
    showFeedback("err", "Request failed");
  } finally {
    btn.classList.remove("busy"); btn.textContent = "▶ Apply HSV";
  }
}

function showFeedback(cls, msg) {
  showFeedbackEl(document.getElementById("fb"), cls, msg);
}
function showFeedbackEl(el, cls, msg) {
  if (!el) return;
  el.className = "feedback " + cls;
  el.textContent = msg;
  setTimeout(() => { el.className = "feedback"; el.textContent = ""; }, 3000);
}

// =============================================================================
// DRAG RESIZE
// =============================================================================

function makeDragDivider(divId, panelId, side) {
  const div   = document.getElementById(divId);
  const panel = document.getElementById(panelId);
  if (!div || !panel) return;

  div.addEventListener("mousedown", e => {
    e.preventDefault();
    div.classList.add("dragging");
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";

    const onMove = ev => {
      const mainRect = document.querySelector("main").getBoundingClientRect();
      let newW;
      if (side === "left") {
        newW = Math.max(220, Math.min(window.innerWidth * 0.45, ev.clientX - mainRect.left));
      } else {
        newW = Math.max(180, Math.min(window.innerWidth * 0.55, mainRect.right - ev.clientX));
      }
      panel.style.width = newW + "px";
    };

    const onUp = () => {
      div.classList.remove("dragging");
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
  });
}

makeDragDivider("dragLeft",  "settingsPanel", "left");
makeDragDivider("dragRight", "consolePanel",  "right");

// =============================================================================
// STATS PARSING
// =============================================================================

const RE_FPS = /FPS:\s*(\d+).*?AI:\s*(\d+).*?CV:\s*(\d+).*?Fused:\s*(\d+).*?Hits:\s*(\d+).*?Possible:\s*(\d+)/;
const RE_CAM = /Camera mode:\s*(\w+)/;

function parseStats(line) {
  let m = RE_FPS.exec(line);
  if (m) {
    [["s-fps",m[1]],["s-ai",m[2]],["s-cv",m[3]],
     ["s-fused",m[4]],["s-hits",m[5]],["s-poss",m[6]]].forEach(([id,v]) => {
      const el = document.getElementById(id);
      el.textContent = v;
      const stat = el.closest(".stat");
      stat.classList.add("flash");
      setTimeout(() => stat.classList.remove("flash"), 400);
    });
  }
  m = RE_CAM.exec(line);
  if (m) {
    document.getElementById("s-cam").textContent    = m[1];
    document.getElementById("camLabel").textContent = "CAM: " + m[1].toUpperCase();
  }
}

// =============================================================================
// CONSOLE
// =============================================================================

function cls(line) {
  if (/FPS:/i.test(line))                                  return "fps";
  if (/verbonden!|connected|verbonden\s*$/i.test(line))    return "ok";
  if (/error|failed|geen.*camera|not avail/i.test(line))   return "err";
  if (/warn/i.test(line))                                  return "warn";
  if (/Dashboard:|http:\/\//i.test(line))                  return "info";
  if (/^──/.test(line.trim()))                             return "sep";
  return "";
}

const logEl = document.getElementById("log");

function addLine(text) {
  if (text === lastText && lastMsgEl) {
    lastCount++;
    let badge = lastMsgEl.querySelector(".repeat-badge");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "repeat-badge";
      lastMsgEl.appendChild(badge);
    }
    badge.textContent = "×" + lastCount;
    if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
    return;
  }

  while (logEl.children.length >= MAX_LINES) logEl.removeChild(logEl.firstChild);

  const ts   = new Date().toTimeString().slice(0, 8);
  const row  = document.createElement("div");
  row.className = "ll";

  const msgSpan = document.createElement("span");
  msgSpan.className = "lm " + cls(text);
  msgSpan.textContent = text;

  row.innerHTML = `<span class="lt">${ts}</span>`;
  row.appendChild(msgSpan);
  logEl.appendChild(row);

  lastText  = text;
  lastCount = 1;
  lastMsgEl = msgSpan;

  lineCount++;
  document.getElementById("lc").textContent = lineCount + " regels";
  if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.innerHTML = "";
  lineCount = 0; lastText = null; lastMsgEl = null;
  document.getElementById("lc").textContent = "0 regels";
}

function toggleScroll() {
  autoScroll = !autoScroll;
  const btn = document.getElementById("scrollBtn");
  btn.textContent = autoScroll ? "↓ Auto" : "⏸ Gepauzeerd";
  btn.className   = autoScroll ? "active" : "";
  if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
}

logEl.addEventListener("scroll", () => {
  const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
  if (!atBottom && autoScroll) {
    autoScroll = false;
    document.getElementById("scrollBtn").textContent = "⏸ Gepauzeerd";
    document.getElementById("scrollBtn").className   = "";
  }
});

// =============================================================================
// LIVE DOT / CONN
// =============================================================================

function setLive(on, text) {
  document.getElementById("liveDot").className = on ? "dot on" : "dot";
  document.getElementById("connDot").className = on ? "dot on" : "dot";
  document.getElementById("liveText").textContent = on ? "LIVE" : text;
  document.getElementById("connText").textContent = text;
}

// =============================================================================
// CLOCK
// =============================================================================

function updateClock() {
  document.getElementById("timeLabel").textContent = new Date().toLocaleTimeString("nl-NL");
}
setInterval(updateClock, 1000);
updateClock();

// =============================================================================
// VIDEO FEED
// =============================================================================

const FEED_RETRY_MS  = 3000;
const FEED_TIMEOUT   = 12000;
const FEED_MAX_RETRY = 15000;

const feed     = document.getElementById("feed");
const noSignal = document.getElementById("noSignal");

let feedAlive      = false;
let feedRetryTimer = null;
let feedWatchdog   = null;
let feedRetryDelay = FEED_RETRY_MS;

function _clearFeedTimers() { clearTimeout(feedRetryTimer); clearTimeout(feedWatchdog); feedRetryTimer = feedWatchdog = null; }

function _armWatchdog() {
  clearTimeout(feedWatchdog);
  feedWatchdog = setTimeout(() => { addLine("── feed watchdog: herverbinden ──"); loadFeed(); }, FEED_TIMEOUT);
}

function loadFeed() {
  _clearFeedTimers();
  requestAnimationFrame(() => {
    feed.src = "/video_feed?" + Date.now();
    feedRetryTimer = setTimeout(() => { if (!feedAlive) onFeedErr(); }, 10000);
  });
}

function onFeedLoad() {
  _clearFeedTimers();
  feedAlive = true; feedRetryDelay = FEED_RETRY_MS;
  feed.style.display = "block"; feed.style.opacity = "1";
  noSignal.style.display = "none";
  _armWatchdog();
}

function onFeedErr() {
  _clearFeedTimers();
  feedAlive = false;
  feed.style.display = "block"; feed.style.opacity = "0.18";
  noSignal.style.display = "flex";
  feedRetryTimer = setTimeout(loadFeed, feedRetryDelay);
  feedRetryDelay = Math.min(feedRetryDelay * 1.5, FEED_MAX_RETRY);
}

feed.addEventListener("load",  onFeedLoad);
feed.addEventListener("error", onFeedErr);
document.addEventListener("visibilitychange", () => { if (!document.hidden) loadFeed(); });
window.addEventListener("focus", () => { if (feedAlive) loadFeed(); });
loadFeed();

// =============================================================================
// SSE
// =============================================================================

function connect() {
  if (evtSrc) { evtSrc.close(); evtSrc = null; }
  evtSrc = new EventSource("/logs");

  evtSrc.onopen = () => {
    setLive(true, "Verbonden met log stream");
    addLine("── log stream verbonden ──");
  };

  evtSrc.onmessage = (e) => { parseStats(e.data); addLine(e.data); };

  evtSrc.onerror = () => {
    setLive(false, `Verbroken – opnieuw over ${RECONNECT_MS / 1000}s…`);
    evtSrc.close(); evtSrc = null;
    setTimeout(connect, RECONNECT_MS);
  };
}

connect();

// =============================================================================
// BOOT
// =============================================================================

initUI();
updateLabels();
</script>
</body>
</html>
"""
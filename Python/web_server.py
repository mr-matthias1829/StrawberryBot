"""Lightweight web dashboard voor de Strawberry Fusion Detector.

Publieke API
-----------
start(host, port)              – start Flask in een daemon-thread (eenmalig)
push_frame(bgr)                – voed een geannoteerd BGR numpy-array aan de MJPEG-stream
push_log(line)                 – stuur een logregel naar alle SSE-clients
set_servo_controller(ctrl)     – koppel de ServoController zodat het dashboard
                                 de speed_scale kan aansturen
"""
from __future__ import annotations

import io
import logging
import queue
import socket
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

# Suppress werkzeug HTTP access logs (127.0.0.1 - - [...] lines)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ── gedeelde staat ─────────────────────────────────────────────────────────────

_app = Flask(__name__)

_frame_lock:   threading.Lock    = threading.Lock()
_latest_jpeg:  bytes | None      = None

_subs_lock:    threading.Lock    = threading.Lock()
_subs:         List[queue.Queue] = []

_started    = False
_start_lock = threading.Lock()

# The ServoController instance — set via set_servo_controller().
# None until the caller provides one; the /servo_speed endpoint
# gracefully returns 503 if it has not been set yet.
_servo_ctrl = None
_servo_ctrl_lock = threading.Lock()


# ── publieke API ───────────────────────────────────────────────────────────────

def push_frame(bgr: np.ndarray) -> None:
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
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


def set_servo_controller(ctrl) -> None:
    """
    Register the ServoController with the web server.

    Must be called before start() (or shortly after) so that the dashboard
    slider can actually change speed_scale on the real controller.

    Parameters
    ----------
    ctrl : ServoController   The live controller instance from dynamixel.py.
    """
    global _servo_ctrl
    with _servo_ctrl_lock:
        _servo_ctrl = ctrl


def _local_ip() -> str:
    """Detecteer het lokale LAN-IP automatisch."""
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

    sys.stdout = _Tee(sys.stdout)

    t = threading.Thread(
        target=lambda: _app.run(
            host=host, port=port,
            threaded=True,
            use_reloader=False,
        ),
        daemon=True,
        name="flask-dashboard",
    )
    t.start()
    time.sleep(0.4)
    print(f"Dashboard: http://{_local_ip()}:{port}/")


# ── stdout interceptor ─────────────────────────────────────────────────────────

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


# ── MJPEG generator ────────────────────────────────────────────────────────────

def _gen_mjpeg():
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(1.0 / 20.0)


# ── Flask routes ───────────────────────────────────────────────────────────────

_NO_CACHE = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma":        "no-cache",
    "Expires":       "0",
    "X-Accel-Buffering": "no",
}


@_app.route("/video_feed")
def route_video():
    resp = Response(
        _gen_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
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


@_app.route("/servo_speed", methods=["GET", "POST"])
def route_servo_speed():
    """
    GET  → returns {"scale": 0.0..1.0} (current speed_scale)
    POST → body {"scale": 0.0..1.0}    (set new speed_scale)

    Called by the dashboard slider. If no ServoController has been registered
    via set_servo_controller(), returns 503.
    """
    with _servo_ctrl_lock:
        ctrl = _servo_ctrl

    if ctrl is None:
        return jsonify({"error": "No servo controller registered"}), 503

    if request.method == "GET":
        return jsonify({"scale": ctrl.speed_scale})

    data = request.get_json(silent=True) or {}
    try:
        scale = float(data["scale"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Expected JSON body: {\"scale\": 0.0..1.0}"}), 400

    ctrl.set_speed_scale(scale)
    print(f"Servo speed scale set to {scale:.0%}")
    return jsonify({"scale": ctrl.speed_scale})


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


# ── Embedded dashboard HTML ────────────────────────────────────────────────────

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
    }

    html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); }
    body { font-family: var(--mono); font-size: 12px; display: flex; flex-direction: column; }

    /* ─── HEADER ─────────────────────────────────────────── */
    header {
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 0 18px;
      height: 52px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      overflow: hidden;
    }

    .brand { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }

    .berry-icon {
      font-size: 20px;
      filter: drop-shadow(0 0 6px rgba(255,61,90,.5));
      animation: float 3s ease-in-out infinite;
    }
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

    /* ─── SERVO SPEED CONTROL ────────────────────────────── */
    .servo-control {
      display: flex; align-items: center; gap: 10px;
      padding: 5px 12px;
      background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
      flex-shrink: 0;
    }
    .servo-control label {
      font-size: 9px; color: var(--muted); text-transform: uppercase;
      letter-spacing: .8px; white-space: nowrap;
    }

    /* Custom slider */
    #speedSlider {
      -webkit-appearance: none;
      appearance: none;
      width: 130px; height: 4px;
      border-radius: 2px;
      outline: none; cursor: pointer;
      background: var(--border2);
    }
    #speedSlider::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px; height: 14px; border-radius: 50%;
      background: var(--orange);
      box-shadow: 0 0 6px rgba(255,140,66,.5);
      cursor: pointer; transition: transform .1s;
    }
    #speedSlider::-webkit-slider-thumb:active { transform: scale(1.3); }
    #speedSlider::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 50%; border: none;
      background: var(--orange); cursor: pointer;
    }

    #speedVal {
      font-family: var(--sans); font-size: 15px; font-weight: 700;
      color: var(--orange); min-width: 36px; text-align: right;
    }
    #speedVal.zero { color: var(--muted); }
    #speedVal.full { color: var(--green); }

    .servo-badge {
      font-size: 9px; padding: 1px 7px; border-radius: 10px;
      background: var(--border); color: var(--muted);
      white-space: nowrap; transition: all .3s;
    }
    .servo-badge.active { background: rgba(255,140,66,.15); border-color: var(--orange); color: var(--orange); }
    .servo-badge.locked { background: var(--green-lo); color: var(--green); }

    .stats { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; align-items: center; }

    .stat {
      display: flex; flex-direction: column; align-items: center;
      padding: 4px 11px 3px;
      background: var(--bg2); border: 1px solid var(--border); border-radius: 7px;
      min-width: 48px; transition: border-color .3s;
    }
    .stat.flash { border-color: var(--green); }

    .stat-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; line-height: 1; }
    .stat-val  { font-family: var(--sans); font-size: 16px; font-weight: 700; line-height: 1.4; color: var(--text); transition: color .2s; }

    #s-fps  { color: var(--green); }
    #s-hits { color: var(--accent); }
    #s-cam  { font-size: 11px; font-family: var(--mono); }

    /* ─── MAIN LAYOUT ─────────────────────────────────────── */
    main { flex: 1; display: flex; min-height: 0; }

    /* ─── VIDEO PANEL ─────────────────────────────────────── */
    .video-panel {
      flex: 1; min-width: 0; position: relative;
      background: var(--bg);
      display: flex; align-items: center; justify-content: center;
      overflow: hidden;
    }

    .video-panel::after {
      content: '';
      position: absolute; inset: 0;
      background: repeating-linear-gradient(to bottom,transparent 0,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);
      pointer-events: none; z-index: 2;
    }

    #feed { max-width: 100%; max-height: 100%; object-fit: contain; display: none; z-index: 1; }

    .no-signal { display: flex; flex-direction: column; align-items: center; gap: 12px; color: var(--muted); z-index: 1; }
    .no-signal-icon { font-size: 56px; opacity: .15; animation: breathe 3s ease-in-out infinite; }
    @keyframes breathe { 0%,100%{opacity:.15} 50%{opacity:.25} }
    .no-signal p { font-size: 11px; letter-spacing: .5px; }

    .corner-label { position: absolute; z-index: 3; font-size: 9px; color: rgba(255,255,255,.25); letter-spacing: .5px; text-transform: uppercase; }
    .corner-label.tl { top: 8px; left: 10px; }
    .corner-label.tr { top: 8px; right: 10px; }
    .corner-label.bl { bottom: 8px; left: 10px; }

    /* ─── DRAG DIVIDER ────────────────────────────────────── */
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
    .lm.servo { color: var(--orange); }

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

    /* ─── RESPONSIVE ──────────────────────────────────────── */
    @media (max-width: 820px) {
      main { flex-direction: column; }
      .video-panel { flex: none; height: 50vh; }
      .drag-divider { width: 100%; height: 5px; cursor: row-resize; }
      .console-panel { width: 100% !important; min-width: unset; }
      .stats { display: none; }
      .servo-control { display: none; }
    }
  </style>
</head>
<body>

<header>
  <div class="brand">
    <div class="berry-icon">🍓</div>
    <h1><span>Strawberry</span> Detector</h1>
  </div>
  <div class="hdivider"></div>
  <div class="live-pill">
    <div class="dot" id="liveDot"></div>
    <span id="liveText">Verbinden…</span>
  </div>
  <div class="hdivider"></div>

  <!-- ── Servo speed control ── -->
  <div class="servo-control">
    <label>Servo Speed</label>
    <input type="range" id="speedSlider" min="0" max="100" value="0" step="1">
    <span id="speedVal" class="zero">0%</span>
    <span class="servo-badge" id="servoBadge">PAUSED</span>
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

  <div class="drag-divider" id="dragDiv"></div>

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

// ── Stats ─────────────────────────────────────────────────────────────────────
const RE_FPS = /FPS:\s*(\d+).*?AI:\s*(\d+).*?CV:\s*(\d+).*?Fused:\s*(\d+).*?Hits:\s*(\d+).*?Possible:\s*(\d+)/;
const RE_CAM = /Camera mode:\s*(\w+)/;

function parseStats(line) {
  let m = RE_FPS.exec(line);
  if (m) {
    [['s-fps',m[1]],['s-ai',m[2]],['s-cv',m[3]],
     ['s-fused',m[4]],['s-hits',m[5]],['s-poss',m[6]]].forEach(([id,v]) => {
      const el = document.getElementById(id);
      el.textContent = v;
      const stat = el.closest('.stat');
      stat.classList.add('flash');
      setTimeout(() => stat.classList.remove('flash'), 400);
    });
  }
  m = RE_CAM.exec(line);
  if (m) {
    document.getElementById('s-cam').textContent = m[1];
    document.getElementById('camLabel').textContent = 'CAM: ' + m[1].toUpperCase();
  }
}

// ── Line classifier ───────────────────────────────────────────────────────────
function cls(line) {
  if (/FPS:/i.test(line))                                  return 'fps';
  if (/verbonden!|connected|verbonden\s*$/i.test(line))    return 'ok';
  if (/error|failed|geen.*camera|not avail/i.test(line))   return 'err';
  if (/warn/i.test(line))                                  return 'warn';
  if (/Dashboard:|http:\/\//i.test(line))                  return 'info';
  if (/servo speed scale/i.test(line))                     return 'servo';
  if (/^──/.test(line.trim()))                             return 'sep';
  return '';
}

// ── Console ───────────────────────────────────────────────────────────────────
const logEl = document.getElementById('log');

function addLine(text) {
  if (text === lastText && lastMsgEl) {
    lastCount++;
    let badge = lastMsgEl.querySelector('.repeat-badge');
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'repeat-badge';
      lastMsgEl.appendChild(badge);
    }
    badge.textContent = '×' + lastCount;
    if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
    return;
  }

  while (logEl.children.length >= MAX_LINES)
    logEl.removeChild(logEl.firstChild);

  const ts  = new Date().toTimeString().slice(0, 8);
  const row = document.createElement('div');
  row.className = 'll';

  const msgSpan = document.createElement('span');
  msgSpan.className = 'lm ' + cls(text);
  msgSpan.textContent = text;

  row.innerHTML = `<span class="lt">${ts}</span>`;
  row.appendChild(msgSpan);
  logEl.appendChild(row);

  lastText   = text;
  lastCount  = 1;
  lastMsgEl  = msgSpan;

  lineCount++;
  document.getElementById('lc').textContent = lineCount + ' regels';
  if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.innerHTML = '';
  lineCount = 0; lastText = null; lastMsgEl = null;
  document.getElementById('lc').textContent = '0 regels';
}

function toggleScroll() {
  autoScroll = !autoScroll;
  const btn = document.getElementById('scrollBtn');
  btn.textContent = autoScroll ? '↓ Auto' : '⏸ Gepauzeerd';
  btn.className   = autoScroll ? 'active' : '';
  if (autoScroll) logEl.scrollTop = logEl.scrollHeight;
}

logEl.addEventListener('scroll', () => {
  const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
  if (!atBottom && autoScroll) {
    autoScroll = false;
    const btn = document.getElementById('scrollBtn');
    btn.textContent = '⏸ Gepauzeerd';
    btn.className = '';
  }
});

// ── Servo speed slider ────────────────────────────────────────────────────────
//
// The slider maps 0–100 (integer %) to 0.0–1.0 sent to POST /servo_speed.
// Debounced: only sends after 120 ms of no movement to avoid flooding the Pi.
// On page load, GETs the current value from the server so a refresh doesn't
// silently reset the scale.

const slider   = document.getElementById('speedSlider');
const speedVal = document.getElementById('speedVal');
const badge    = document.getElementById('servoBadge');

let sliderTimer = null;

function updateSliderUI(pct) {
  speedVal.textContent = pct + '%';
  speedVal.className = pct === 0 ? 'zero' : pct === 100 ? 'full' : '';
  if (pct === 0) {
    badge.textContent = 'PAUSED';
    badge.className   = 'servo-badge';
  } else if (pct === 100) {
    badge.textContent = 'FULL';
    badge.className   = 'servo-badge locked';
  } else {
    badge.textContent = pct + '%';
    badge.className   = 'servo-badge active';
  }
  // Colour the slider track fill
  slider.style.background = `linear-gradient(to right, var(--orange) ${pct}%, var(--border2) ${pct}%)`;
}

async function sendSpeedScale(scale) {
  try {
    const r = await fetch('/servo_speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scale }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      addLine('⚠ Servo speed error: ' + (j.error || r.status));
    }
  } catch (e) {
    addLine('⚠ Servo speed request failed: ' + e.message);
  }
}

slider.addEventListener('input', () => {
  const pct = parseInt(slider.value, 10);
  updateSliderUI(pct);
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => sendSpeedScale(pct / 100), 120);
});

// On load: fetch current scale from server
async function fetchCurrentScale() {
  try {
    const r = await fetch('/servo_speed');
    if (r.ok) {
      const j = await r.json();
      const pct = Math.round((j.scale || 0) * 100);
      slider.value = pct;
      updateSliderUI(pct);
    }
  } catch (_) { /* server not ready yet, stay at 0 */ }
}
fetchCurrentScale();
updateSliderUI(0); // set initial gradient before fetch returns

// ── Drag-to-resize ────────────────────────────────────────────────────────────
const dragDiv    = document.getElementById('dragDiv');
const consolePnl = document.getElementById('consolePanel');

dragDiv.addEventListener('mousedown', e => {
  e.preventDefault();
  dragDiv.classList.add('dragging');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';

  const onMove = ev => {
    const mainRect = document.querySelector('main').getBoundingClientRect();
    const newW = Math.max(180, Math.min(window.innerWidth * 0.7, mainRect.right - ev.clientX));
    consolePnl.style.width = newW + 'px';
    document.documentElement.style.setProperty('--console-w', newW + 'px');
  };

  const onUp = () => {
    dragDiv.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  };

  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
});

// ── Live indicator ────────────────────────────────────────────────────────────
function setLive(on, text) {
  document.getElementById('liveDot').className = on ? 'dot on' : 'dot';
  document.getElementById('connDot').className = on ? 'dot on' : 'dot';
  document.getElementById('liveText').textContent = on ? 'LIVE' : text;
  document.getElementById('connText').textContent = text;
}

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  document.getElementById('timeLabel').textContent = new Date().toLocaleTimeString('nl-NL');
}
setInterval(updateClock, 1000);
updateClock();

// ── Video feed ────────────────────────────────────────────────────────────────
const FEED_RETRY_MS  = 3000;
const FEED_TIMEOUT   = 12000;
const FEED_MAX_RETRY = 15000;

const feed     = document.getElementById('feed');
const noSignal = document.getElementById('noSignal');

let feedAlive      = false;
let feedRetryTimer = null;
let feedWatchdog   = null;
let feedRetryDelay = FEED_RETRY_MS;

function _clearFeedTimers() {
  clearTimeout(feedRetryTimer);
  clearTimeout(feedWatchdog);
  feedRetryTimer = null;
  feedWatchdog   = null;
}

function _armWatchdog() {
  clearTimeout(feedWatchdog);
  feedWatchdog = setTimeout(() => {
    addLine('── feed watchdog: stream stilgevallen, herverbinden ──');
    loadFeed();
  }, FEED_TIMEOUT);
}

function loadFeed() {
  _clearFeedTimers();
  feed.src = '';
  requestAnimationFrame(() => {
    feed.src = '/video_feed?' + Date.now();
    feedRetryTimer = setTimeout(() => { if (!feedAlive) onFeedErr(); }, 10000);
  });
}

function onFeedLoad() {
  _clearFeedTimers();
  feedAlive      = true;
  feedRetryDelay = FEED_RETRY_MS;
  feed.style.display     = 'block';
  noSignal.style.display = 'none';
  _armWatchdog();
}

function onFeedErr() {
  _clearFeedTimers();
  feedAlive = false;
  feed.style.display     = 'none';
  noSignal.style.display = 'flex';
  feedRetryTimer = setTimeout(loadFeed, feedRetryDelay);
  feedRetryDelay = Math.min(feedRetryDelay * 1.5, FEED_MAX_RETRY);
}

feed.addEventListener('load',  onFeedLoad);
feed.addEventListener('error', onFeedErr);

document.addEventListener('visibilitychange', () => { if (!document.hidden) loadFeed(); });
window.addEventListener('focus', () => { if (feedAlive) loadFeed(); });

loadFeed();

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect() {
  if (evtSrc) { evtSrc.close(); evtSrc = null; }
  evtSrc = new EventSource('/logs');

  evtSrc.onopen = () => {
    setLive(true, 'Verbonden met log stream');
    addLine('── log stream verbonden ──');
  };

  evtSrc.onmessage = (e) => {
    parseStats(e.data);
    addLine(e.data);
  };

  evtSrc.onerror = () => {
    setLive(false, `Verbroken – opnieuw over ${RECONNECT_MS / 1000}s…`);
    evtSrc.close(); evtSrc = null;
    setTimeout(connect, RECONNECT_MS);
  };
}

connect();
</script>
</body>
</html>
"""
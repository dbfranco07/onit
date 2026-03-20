"""
TurtleBot3 ROS 2 Bridge — singleton that manages all ROS 2 communication.

Provides thread-safe access to:
  - Camera (CompressedImage subscriber)
  - LiDAR (LaserScan subscriber)
  - Odometry (Odometry subscriber — for dead reckoning)
  - Velocity commands (Twist publisher on /cmd_vel)

Usage:
    bridge = TurtleBot3Bridge.get_instance()
    image_bytes, ts = bridge.get_image()
    bridge.move_forward(distance_m=0.5, speed=0.2)
"""

import math
import json
import os
import sys
import time
import shutil
import subprocess
import threading
import logging
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from io import BytesIO
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _stderr(msg: str):
    """Print diagnostic to stderr so it isn't swallowed by multiprocessing."""
    print(f"[TurtleBot3Bridge] {msg}", file=sys.stderr, flush=True)


def _open_url_in_browser(url: str) -> bool:
    """Open URL with preferred Chromium launchers, fallback to system browser."""
    candidates = [
        ["flatpak", "run", "org.chromium.Chromium", url],
        ["chromium-browser", url],
        ["chromium", url],
    ]

    for command in candidates:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            continue

    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False

# TurtleBot3 Burger physical limits (ROBOTIS spec)
MAX_LINEAR_SPEED = 0.22   # m/s
MAX_ANGULAR_SPEED = 2.84  # rad/s

# Default timeout for motion commands
DEFAULT_MOTION_TIMEOUT = 30.0  # seconds

# Control loop rate — 20 Hz gives tighter feedback and smoother ramping
CONTROL_HZ = 20  # Hz

# Trapezoidal velocity profile parameters
LINEAR_RAMP_DISTANCE = 0.05   # metres — ramp over first/last 5 cm
ANGULAR_RAMP_DEG = 5.0        # degrees — ramp over first/last 5°
MIN_LINEAR_SPEED = 0.03       # m/s — floor so robot doesn't stall
MIN_ANGULAR_SPEED = 0.1       # rad/s — floor so robot doesn't stall
LINEAR_CMD_SMOOTHING_ALPHA = 0.35   # 0..1; higher follows target faster
LINEAR_ACCEL_LIMIT = 0.35           # m/s^2 max increase per second
LINEAR_DECEL_LIMIT = 0.6            # m/s^2 max decrease per second

# Default obstacle safety margin
DEFAULT_SAFETY_MARGIN_M = 0.25

# Default camera viewer port
DEFAULT_VIEWER_PORT = 18280

# LiDAR orientation calibration (degrees).
# Use this when sensor-frame 0° is not aligned with robot forward in the UI.
# Negative rotates clockwise, positive rotates counter-clockwise.
LIDAR_FRAME_ROTATION_DEG = -90.0

# Progressive scan performance tuning
MAX_SEARCH_FRAMES = 36
SEARCH_FRAME_MIN_HEADING_DELTA_DEG = 4.0
MOSAIC_REBUILD_MIN_INTERVAL_S = 0.5

# -----------------------------------------------------------------------
# MJPEG Camera Viewer — streams live camera feed to the browser
# -----------------------------------------------------------------------

_MOSAIC_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>TurtleBot3 Panoramic Mosaic</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111; color: #ddd;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; min-height: 100vh; padding: 12px;
    }
    .header {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 8px; flex-wrap: wrap;
      justify-content: center;
    }
    h1 { font-size: 1.2em; color: #fff; white-space: nowrap; }
    .stats {
      font-size: 0.85em; color: #aaa;
      display: flex; gap: 16px;
    }
    .stats span { white-space: nowrap; }
    nav { margin-bottom: 10px; }
    nav a {
      color: #7cb3ff; text-decoration: none;
      padding: 6px 14px; border: 1px solid #555;
      background: #222; border-radius: 4px;
      font-size: 0.85em;
    }
    nav a:hover { background: #333; border-color: #888; }
    .controls {
      display: flex; gap: 8px; margin-bottom: 10px;
    }
    .controls button {
      padding: 6px 14px; border: 1px solid #555;
      background: #222; color: #ddd; border-radius: 4px;
      cursor: pointer; font-size: 0.85em;
    }
    .controls button:hover { background: #333; border-color: #888; }
    .feed-container {
      position: relative;
      width: 100%; max-width: 1400px;
      display: flex; justify-content: center;
    }
    #mosaic {
      width: 100%; max-height: 85vh;
      object-fit: contain;
      border: 2px solid #333; border-radius: 6px;
      background: #000;
    }
    .placeholder {
      color: #666; font-size: 1.1em;
      text-align: center; padding: 60px 20px;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>&#x1F5BC; Panoramic Mosaic</h1>
    <div class="stats">
      <span id="status">Waiting for scan&hellip;</span>
      <span id="ts"></span>
    </div>
  </div>
  <nav><a href="/">&#127909; Live Camera</a></nav>
  <div class="controls">
    <button onclick="refresh()" title="Refresh mosaic (R)">&#x1F504; Refresh</button>
    <button onclick="toggleAuto()" id="autoBtn" title="Toggle auto-refresh (A)">&#x23F1; Auto: ON (3s)</button>
    <button onclick="download()" title="Download mosaic (D)">&#x1F4BE; Download</button>
  </div>
  <div class="feed-container">
    <div id="placeholderBox" class="placeholder">No frames captured yet. The mosaic builds progressively as the robot searches.</div>
    <img id="mosaic" style="display:none" alt="Panoramic Mosaic" />
  </div>
  <script>
    var img = document.getElementById('mosaic');
    var ph  = document.getElementById('placeholderBox');
    var st  = document.getElementById('status');
    var tsEl = document.getElementById('ts');
    var autoOn = true, timer = null;

    function refresh() {
      fetch('/mosaic?' + Date.now()).then(function(r) {
        if (r.ok) {
          var stamp = r.headers.get('X-Mosaic-Stamp') || '';
          return r.blob().then(function(b) { return { blob: b, stamp: stamp }; });
        }
        throw new Error(r.status);
      }).then(function(result) {
        img.src = URL.createObjectURL(result.blob);
        img.style.display = 'block';
        ph.style.display = 'none';
        st.textContent = 'Available';
        if (result.stamp) tsEl.textContent = result.stamp;
      }).catch(function() {
        st.textContent = 'No mosaic yet';
      });
    }

    function toggleAuto() {
      autoOn = !autoOn;
      document.getElementById('autoBtn').textContent =
        autoOn ? '\\u23F1 Auto: ON (3s)' : '\\u23F1 Auto: OFF';
      if (autoOn) startTimer(); else stopTimer();
    }
    function startTimer() { stopTimer(); timer = setInterval(refresh, 3000); }
    function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

    function download() {
      if (img.src) {
        var a = document.createElement('a');
        a.href = '/mosaic'; a.download = 'mosaic_' + Date.now() + '.jpg';
        a.click();
      }
    }

    document.addEventListener('keydown', function(e) {
      if (e.key === 'r' || e.key === 'R') refresh();
      if (e.key === 'a' || e.key === 'A') toggleAuto();
      if (e.key === 'd' || e.key === 'D') download();
    });

    refresh();
    startTimer();
  </script>
</body>
</html>
"""

_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>TurtleBot3 Camera Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #111; color: #ddd;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh; padding: 12px;
    }
    .header {
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 8px; flex-wrap: wrap;
      justify-content: center;
    }
    h1 { font-size: 1.2em; color: #fff; white-space: nowrap; }
    .stats {
      font-size: 0.85em; color: #aaa;
      display: flex; gap: 16px;
    }
    .stats span { white-space: nowrap; }
    nav { margin-bottom: 10px; }
    nav a {
      color: #7cb3ff; text-decoration: none;
      padding: 6px 14px; border: 1px solid #555;
      background: #222; border-radius: 4px;
      font-size: 0.85em;
    }
    nav a:hover { background: #333; border-color: #888; }
    .controls {
      display: flex; gap: 8px; margin-bottom: 10px;
    }
    .controls button {
      padding: 6px 14px; border: 1px solid #555;
      background: #222; color: #ddd; border-radius: 4px;
      cursor: pointer; font-size: 0.85em;
    }
    .controls button:hover { background: #333; border-color: #888; }
    .feed-container {
      position: relative;
            width: 100%; max-width: 920px;
      display: flex; justify-content: center;
    }
        .layout {
            width: 100%; max-width: 1500px;
            display: flex; gap: 14px;
            align-items: flex-start;
            justify-content: center;
        }
        .panel {
            width: min(430px, 96vw);
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 6px;
            padding: 10px;
            font-size: 0.86em;
            line-height: 1.35;
        }
        .panel h2 {
            font-size: 1em;
            margin-bottom: 8px;
            color: #fff;
        }
        .overview {
            color: #c9e6ff;
            margin-bottom: 10px;
            min-height: 38px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 10px;
            margin-bottom: 10px;
        }
        .k { color: #9aa0a6; }
        .v { color: #e8eaed; }
        .subsection {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #2f2f2f;
        }
        .plan-wrap {
            width: 100%;
            background: #101214;
            border: 1px solid #2f2f2f;
            border-radius: 4px;
            padding: 6px;
        }
        #planCanvas {
            width: 100%;
            height: 200px;
            border-radius: 4px;
            display: block;
            background: #0d1013;
        }
        .plan-meta {
            margin-top: 6px;
            color: #9aa0a6;
            font-size: 0.9em;
        }
        ul.logs {
            list-style: none;
            margin: 6px 0 0;
            padding: 0;
            max-height: 310px;
            overflow: auto;
        }
        ul.logs li {
            border: 1px solid #2f2f2f;
            border-radius: 4px;
            padding: 6px;
            margin-bottom: 6px;
            background: #151515;
        }
        .tool { color: #ffd580; font-weight: 600; }
        .ts { color: #9aa0a6; font-size: 0.9em; }
        .reason { color: #d0e8d0; }
        .args { color: #b6c3d1; font-size: 0.92em; }
    #cam {
      width: 100%; max-height: 80vh;
      object-fit: contain;
      border: 2px solid #333; border-radius: 6px;
      image-rendering: auto;
      background: #000;
    }
    .live-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #555; display: inline-block;
    }
    .live-dot.active { background: #4caf50; animation: pulse 1.5s infinite; }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
        .fullscreen .feed-container { max-width: none; }
    .fullscreen #cam { max-height: 100vh; border: none; border-radius: 0; }
        @media (max-width: 1200px) {
            .layout { flex-direction: column; align-items: center; }
            .panel { width: min(920px, 96vw); }
        }
  </style>
</head>
<body>
  <div class="header">
    <h1><span class="live-dot" id="liveDot"></span> TurtleBot3 Camera</h1>
    <div class="stats">
      <span id="status">Connecting&hellip;</span>
      <span id="fps"></span>
      <span id="res"></span>
    </div>
  </div>
  <nav><a href="/mosaic.html">&#x1F5BC; Panoramic Mosaic</a></nav>
  <div class="controls">
    <button onclick="takeSnapshot()" title="Save current frame (S)">&#128247; Snapshot</button>
    <button onclick="toggleFullscreen()" title="Toggle fullscreen (F)">&#x26F6; Fullscreen</button>
  </div>
    <div class="layout">
        <div class="feed-container">
            <img id="cam" src="/stream" alt="Camera feed" />
        </div>
        <aside class="panel">
            <h2>Runtime Overview</h2>
            <div id="overview" class="overview">Waiting for status…</div>

            <div class="subsection">
                <h2>Pose</h2>
                <div class="grid">
                    <div class="k">X</div><div class="v" id="poseX">—</div>
                    <div class="k">Y</div><div class="v" id="poseY">—</div>
                    <div class="k">Heading</div><div class="v" id="poseHeading">—</div>
                    <div class="k">Linear</div><div class="v" id="poseLinear">—</div>
                    <div class="k">Angular</div><div class="v" id="poseAngular">—</div>
                </div>
            </div>

            <div class="subsection">
                <h2>LiDAR Summary</h2>
                <div class="grid">
                    <div class="k">Front min</div><div class="v" id="lidarFront">—</div>
                    <div class="k">Left min</div><div class="v" id="lidarLeft">—</div>
                    <div class="k">Right min</div><div class="v" id="lidarRight">—</div>
                    <div class="k">Back min</div><div class="v" id="lidarBack">—</div>
                </div>
            </div>

            <div class="subsection">
                <h2>Plan Position</h2>
                <div class="plan-wrap">
                    <canvas id="planCanvas" width="400" height="200"></canvas>
                    <div class="plan-meta" id="planMeta">Waiting for odometry…</div>
                </div>
            </div>

            <div class="subsection">
                <h2>Recent Tool Calls</h2>
                <ul id="toolLogs" class="logs"></ul>
            </div>
        </aside>
  </div>
  <script>
    var img = document.getElementById('cam');
    var st  = document.getElementById('status');
    var fpsSp = document.getElementById('fps');
    var resSp = document.getElementById('res');
    var dot = document.getElementById('liveDot');
    var frames = 0, lastTime = performance.now(), fpsVal = 0;

    var overviewEl = document.getElementById('overview');
    var poseXEl = document.getElementById('poseX');
    var poseYEl = document.getElementById('poseY');
    var poseHeadingEl = document.getElementById('poseHeading');
    var poseLinearEl = document.getElementById('poseLinear');
    var poseAngularEl = document.getElementById('poseAngular');

    var lidarFrontEl = document.getElementById('lidarFront');
    var lidarLeftEl = document.getElementById('lidarLeft');
    var lidarRightEl = document.getElementById('lidarRight');
    var lidarBackEl = document.getElementById('lidarBack');

    var toolLogsEl = document.getElementById('toolLogs');
    var planCanvas = document.getElementById('planCanvas');
    var planCtx = planCanvas.getContext('2d');
    var planMetaEl = document.getElementById('planMeta');
    var planTrail = [];

    img.onload = function() {
      frames++;
      var now = performance.now();
      var dt = (now - lastTime) / 1000;
      if (dt >= 1.0) {
        fpsVal = Math.round(frames / dt);
        fpsSp.textContent = fpsVal + ' fps';
        frames = 0; lastTime = now;
      }
      st.textContent = 'Live';
      dot.classList.add('active');
      if (img.naturalWidth) resSp.textContent = img.naturalWidth + '\\u00d7' + img.naturalHeight;
    };
    img.onerror = function() {
      st.textContent = 'Reconnecting\\u2026';
      dot.classList.remove('active');
      setTimeout(function() { img.src = '/stream?' + Date.now(); }, 1500);
    };

    function takeSnapshot() {
      var a = document.createElement('a');
      a.href = '/snapshot'; a.download = 'turtlebot3_' + Date.now() + '.jpg';
      a.click();
    }
    function toggleFullscreen() {
      if (!document.fullscreenElement) {
        document.body.requestFullscreen().then(function() { document.body.classList.add('fullscreen'); });
      } else {
        document.exitFullscreen().then(function() { document.body.classList.remove('fullscreen'); });
      }
    }
    document.addEventListener('keydown', function(e) {
      if (e.key === 'f' || e.key === 'F') toggleFullscreen();
      if (e.key === 's' || e.key === 'S') takeSnapshot();
    });

        function fmtM(v) {
            if (v === null || v === undefined) return '—';
            return v.toFixed ? v.toFixed(2) + ' m' : String(v) + ' m';
        }

        function updateStatusPanel(status) {
            overviewEl.textContent = status.overview || 'No overview available';

            var pose = status.pose || {};
            poseXEl.textContent = pose.x_m !== undefined ? pose.x_m.toFixed(3) + ' m' : '—';
            poseYEl.textContent = pose.y_m !== undefined ? pose.y_m.toFixed(3) + ' m' : '—';
            poseHeadingEl.textContent = pose.heading_deg !== undefined ? pose.heading_deg.toFixed(1) + '°' : '—';
            poseLinearEl.textContent = pose.linear_m_s !== undefined ? pose.linear_m_s.toFixed(3) + ' m/s' : '—';
            poseAngularEl.textContent = pose.angular_rad_s !== undefined ? pose.angular_rad_s.toFixed(3) + ' rad/s' : '—';
            updatePlanPosition(pose, status.lidar || {});

            var sectors = (((status.lidar || {}).sectors) || {});
            lidarFrontEl.textContent = fmtM((sectors.front || {}).min_m);
            lidarLeftEl.textContent = fmtM((sectors.left || {}).min_m);
            lidarRightEl.textContent = fmtM((sectors.right || {}).min_m);
            lidarBackEl.textContent = fmtM((sectors.back || {}).min_m);

            var logs = status.recent_tools || [];
            if (!logs.length) {
                toolLogsEl.innerHTML = '<li><span class="reason">No tool calls yet.</span></li>';
                return;
            }
            toolLogsEl.innerHTML = logs.slice().reverse().map(function(item) {
                var args = item.args && Object.keys(item.args).length ? JSON.stringify(item.args) : '{}';
                var argWhy = item.arg_reasons && Object.keys(item.arg_reasons).length
                    ? JSON.stringify(item.arg_reasons)
                    : '{}';
                return '<li>' +
                    '<div><span class="tool">' + item.tool + '</span> <span class="ts">' + (item.timestamp || '') + '</span></div>' +
                    '<div class="reason">' + (item.reason || 'No reason provided') + '</div>' +
                    '<div class="args">args: ' + args + '</div>' +
                    '<div class="args">arg why: ' + argWhy + '</div>' +
                    '</li>';
            }).join('');
        }

        function drawArrow(ctx, x, y, angleRad, color) {
            var len = 16;
            var hx = x + Math.cos(angleRad) * len;
            var hy = y - Math.sin(angleRad) * len;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(hx, hy);
            ctx.stroke();

            var head = 6;
            var a1 = angleRad + Math.PI * 0.82;
            var a2 = angleRad - Math.PI * 0.82;
            ctx.beginPath();
            ctx.moveTo(hx, hy);
            ctx.lineTo(hx + Math.cos(a1) * head, hy - Math.sin(a1) * head);
            ctx.lineTo(hx + Math.cos(a2) * head, hy - Math.sin(a2) * head);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
        }

        function updatePlanPosition(pose, lidar) {
            if (pose.x_m === undefined || pose.y_m === undefined) {
                return;
            }

            var x = Number(pose.x_m);
            var y = Number(pose.y_m);
            var headingDeg = Number(pose.heading_deg || 0);
            var headingRad = headingDeg * Math.PI / 180.0;
            // Display rotation: rotate map 90° so 0° heading is visually "up".
            var displayRotationRad = Math.PI / 2.0;

            if (!planTrail.length) {
                planTrail.push({x: x, y: y});
            } else {
                var last = planTrail[planTrail.length - 1];
                var dx = x - last.x;
                var dy = y - last.y;
                if (Math.hypot(dx, dy) > 0.01) {
                    planTrail.push({x: x, y: y});
                }
            }
            if (planTrail.length > 300) {
                planTrail = planTrail.slice(-300);
            }

            var pad = 18;
            var w = planCanvas.width;
            var h = planCanvas.height;

            var xs = planTrail.map(function(p) { return p.x; });
            var ys = planTrail.map(function(p) { return p.y; });
            var minX = Math.min.apply(null, xs);
            var maxX = Math.max.apply(null, xs);
            var minY = Math.min.apply(null, ys);
            var maxY = Math.max.apply(null, ys);

            var lidarPoints = (lidar && lidar.points) ? lidar.points : [];

            // Expand map bounds to include current LiDAR returns around robot.
            if (lidarPoints.length > 0) {
                var minLX = Infinity, maxLX = -Infinity;
                var minLY = Infinity, maxLY = -Infinity;
                for (var li = 0; li < lidarPoints.length; li++) {
                    var lp = lidarPoints[li];
                    if (lp && lp.world_x_m !== undefined && lp.world_y_m !== undefined) {
                        minLX = Math.min(minLX, lp.world_x_m);
                        maxLX = Math.max(maxLX, lp.world_x_m);
                        minLY = Math.min(minLY, lp.world_y_m);
                        maxLY = Math.max(maxLY, lp.world_y_m);
                    }
                }
                if (minLX !== Infinity) {
                    minX = Math.min(minX, minLX);
                    maxX = Math.max(maxX, maxLX);
                    minY = Math.min(minY, minLY);
                    maxY = Math.max(maxY, maxLY);
                }
            }

            var spanX = Math.max(0.6, maxX - minX);
            var spanY = Math.max(0.6, maxY - minY);
            var span = Math.max(spanX, spanY);

            var cx = (minX + maxX) / 2;
            var cy = (minY + maxY) / 2;

            function toPx(px, py) {
                var dx = px - cx;
                var dy = py - cy;

                // Rotate the entire plan view for operator-friendly orientation.
                var rx = dx * Math.cos(displayRotationRad) - dy * Math.sin(displayRotationRad);
                var ry = dx * Math.sin(displayRotationRad) + dy * Math.cos(displayRotationRad);

                var nx = (rx + span / 2) / span;
                var ny = (ry + span / 2) / span;
                return {
                    x: pad + nx * (w - 2 * pad),
                    y: h - (pad + ny * (h - 2 * pad))
                };
            }

            planCtx.clearRect(0, 0, w, h);

            planCtx.strokeStyle = '#1f2a34';
            planCtx.lineWidth = 1;
            for (var gx = 0; gx <= 4; gx++) {
                var xx = pad + (gx / 4) * (w - 2 * pad);
                planCtx.beginPath();
                planCtx.moveTo(xx, pad);
                planCtx.lineTo(xx, h - pad);
                planCtx.stroke();
            }
            for (var gy = 0; gy <= 4; gy++) {
                var yy = pad + (gy / 4) * (h - 2 * pad);
                planCtx.beginPath();
                planCtx.moveTo(pad, yy);
                planCtx.lineTo(w - pad, yy);
                planCtx.stroke();
            }

            if (planTrail.length >= 2) {
                planCtx.strokeStyle = '#65d6ff';
                planCtx.lineWidth = 2;
                planCtx.beginPath();
                var p0 = toPx(planTrail[0].x, planTrail[0].y);
                planCtx.moveTo(p0.x, p0.y);
                for (var i = 1; i < planTrail.length; i++) {
                    var pi = toPx(planTrail[i].x, planTrail[i].y);
                    planCtx.lineTo(pi.x, pi.y);
                }
                planCtx.stroke();
            }

            // Sonar-like obstacle blips from LiDAR returns.
            if (lidarPoints.length > 0) {
                planCtx.fillStyle = '#73ff9f';
                planCtx.strokeStyle = 'rgba(115,255,159,0.22)';
                planCtx.lineWidth = 1;
                for (var lj = 0; lj < lidarPoints.length; lj++) {
                    var point = lidarPoints[lj];
                    if (!point || point.world_x_m === undefined || point.world_y_m === undefined) {
                        continue;
                    }
                    var pxy = toPx(point.world_x_m, point.world_y_m);
                    planCtx.beginPath();
                    planCtx.arc(pxy.x, pxy.y, 1.8, 0, Math.PI * 2);
                    planCtx.fill();
                }
            }

            var cur = toPx(x, y);
            planCtx.beginPath();
            planCtx.arc(cur.x, cur.y, 5, 0, Math.PI * 2);
            planCtx.fillStyle = '#ffcc66';
            planCtx.fill();
            drawArrow(planCtx, cur.x, cur.y, headingRad + displayRotationRad, '#ffcc66');

            planMetaEl.textContent =
                'Position: (' + x.toFixed(2) + ', ' + y.toFixed(2) + ') m | Heading: ' + headingDeg.toFixed(1) + '° | Trail: ' + planTrail.length + ' | LiDAR blips: ' + lidarPoints.length;
        }

        function refreshStatus() {
            fetch('/status?' + Date.now()).then(function(r) {
                if (!r.ok) throw new Error('status ' + r.status);
                return r.json();
            }).then(updateStatusPanel).catch(function() {
                overviewEl.textContent = 'Status unavailable';
            });
        }

        refreshStatus();
        setInterval(refreshStatus, 1200);
  </script>
</body>
</html>
"""


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new thread.

    Required because the MJPEG /stream endpoint keeps the connection
    open indefinitely, which would block all other routes (like /mosaic)
    on a single-threaded server.
    """
    daemon_threads = True


class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves an HTML page at / and MJPEG stream at /stream."""

    # Set by the factory function
    bridge_ref = None

    def log_message(self, format, *args):  # noqa: A002
        # Silence per-request logs
        pass

    def do_GET(self):
        parsed_path = urlparse(self.path).path

        if parsed_path == '/' or parsed_path.startswith('/index'):
            self._serve_html()
        elif parsed_path.startswith('/stream'):
            self._serve_mjpeg()
        elif parsed_path == '/snapshot':
            self._serve_snapshot()
        elif parsed_path == '/mosaic.html':
            self._serve_mosaic_html()
        elif parsed_path.startswith('/mosaic'):
            self._serve_mosaic_image()
        elif parsed_path == '/status':
            self._serve_status_json()
        else:
            self.send_error(404)

    def _serve_html(self):
        body = _VIEWER_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_snapshot(self):
        bridge = self.bridge_ref
        if bridge is None:
            self.send_error(503, 'Bridge not ready')
            return
        data, _ = bridge.get_image()
        if data is None:
            self.send_error(503, 'No image available')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_mosaic_html(self):
        body = _MOSAIC_HTML.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mosaic_image(self):
        bridge = self.bridge_ref
        if bridge is None:
            self.send_error(503, 'Bridge not ready')
            return
        with bridge._mosaic_lock:
            data = bridge._mosaic_data
            stamp = bridge._mosaic_stamp
        if data is None:
            self.send_error(404, 'No mosaic available yet - search has not started')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Mosaic-Stamp', str(stamp) if stamp else '')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    def _serve_status_json(self):
        bridge = self.bridge_ref
        if bridge is None:
            self.send_error(503, 'Bridge not ready')
            return

        payload = bridge.get_operator_status()
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

        bridge = self.bridge_ref
        prev_stamp = None
        try:
            while True:
                if bridge is None:
                    time.sleep(0.1)
                    continue
                data, stamp = bridge.get_image()
                if data is not None and stamp != prev_stamp:
                    prev_stamp = stamp
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                time.sleep(0.05)  # ~20 fps cap
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected


def _euler_from_quaternion(x, y, z, w):
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Only yaw is needed for the TurtleBot3 (2D planar robot), but we
    return all three for completeness.  No external dependency required.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _normalize_angle(angle_rad):
    """Normalize angle to [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


class TurtleBot3Bridge:
    """Singleton bridge between onit MCP tools and ROS 2 TurtleBot3 topics."""

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        camera_topic="/camera/image/compressed",
        lidar_topic="/scan",
        odom_topic="/odom",
        cmd_vel_topic="/cmd_vel",
        ros_domain_id=None,
    ):
        if TurtleBot3Bridge._instance is not None:
            raise RuntimeError(
                "Use TurtleBot3Bridge.get_instance() instead of direct construction."
            )

        self._camera_topic = camera_topic
        self._lidar_topic = lidar_topic
        self._odom_topic = odom_topic
        self._cmd_vel_topic = cmd_vel_topic

        # Set ROS_DOMAIN_ID before rclpy.init()
        if ros_domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)

        _stderr(f"Initializing — ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(not set)')}")
        _stderr(f"Topics: camera={camera_topic} lidar={lidar_topic} odom={odom_topic}")

        # Late imports — rclpy is only available in a ROS 2 environment
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

        if not rclpy.ok():
            rclpy.init()
            _stderr("rclpy.init() called successfully")
        else:
            _stderr("rclpy already initialized")

        self._node = Node("onit_turtlebot3")
        self._rclpy = rclpy  # stash for spin loop

        # Cached latest messages (protected by locks)
        self._image_lock = threading.Lock()
        self._image_data = None  # bytes
        self._image_stamp = None  # datetime

        self._lidar_lock = threading.Lock()
        self._lidar_data = None  # dict

        self._odom_lock = threading.Lock()
        self._odom_data = None  # dict

        self._shutdown = False  # signal spin thread to stop

        # Camera viewer state
        self._viewer_server = None
        self._viewer_port = None
        self._viewer_thread = None

        # Latest panoramic mosaic (built progressively from search frames)
        self._mosaic_lock = threading.Lock()
        self._mosaic_data = None   # bytes (JPEG)
        self._mosaic_stamp = None  # datetime

        # Progressive search frame accumulator (fed by get_camera_image)
        self._search_frames_lock = threading.Lock()
        self._search_frames: list[tuple[bytes, float]] = []  # [(jpeg, heading_deg), ...]
        self._last_mosaic_rebuild_monotonic = 0.0
        self._mosaic_rebuild_pending = False

        # Lightweight performance metrics (for profiling/tuning)
        self._perf_lock = threading.Lock()
        self._perf_stats = {
            "mosaic_rebuild_count": 0,
            "mosaic_rebuild_avg_ms": 0.0,
            "mosaic_rebuild_last_ms": 0.0,
            "search_frames_accepted": 0,
            "search_frames_skipped": 0,
        }

        self._tool_trace_lock = threading.Lock()
        self._tool_trace: list[dict] = []
        self._tool_trace_max = 80

        # QoS for sensor topics (best-effort — matches typical LiDAR/camera publishers)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # QoS for odom — TurtleBot3 publishes odom with RELIABLE QoS
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        from sensor_msgs.msg import CompressedImage, LaserScan
        from nav_msgs.msg import Odometry

        self._node.create_subscription(
            CompressedImage, self._camera_topic, self._camera_cb, sensor_qos
        )
        self._node.create_subscription(
            LaserScan, self._lidar_topic, self._lidar_cb, sensor_qos
        )
        self._node.create_subscription(
            Odometry, self._odom_topic, self._odom_cb, odom_qos
        )

        # Publisher
        from geometry_msgs.msg import Twist

        self._cmd_vel_pub = self._node.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._Twist = Twist  # stash class ref for later use

        # Spin the node in a daemon thread
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        logger.info(
            "TurtleBot3Bridge initialized — camera=%s  lidar=%s  odom=%s  cmd_vel=%s",
            camera_topic, lidar_topic, odom_topic, cmd_vel_topic,
        )

        # Wait for DDS discovery and first messages (up to 10s)
        self._warmup(timeout=10.0)

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, **kwargs):
        """Return the singleton instance, creating it on first call.

        Keyword arguments are forwarded to ``__init__`` only on first call.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Destroy the singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance._shutdown = True
                    cls._instance._node.destroy_node()
                except Exception:
                    pass
                cls._instance = None

    # ------------------------------------------------------------------
    # ROS 2 spin
    # ------------------------------------------------------------------

    def _spin(self):
        """Spin the ROS 2 node in a loop using spin_once.

        Using spin_once() instead of blocking spin() is more robust in
        daemon threads and allows graceful shutdown.
        """
        try:
            while not self._shutdown and self._rclpy.ok():
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
        except Exception:
            logger.exception("ROS 2 spin interrupted")

    def _warmup(self, timeout=10.0):
        """Wait for DDS discovery — block until odom or scan is received."""
        _stderr(f"Warmup started (timeout={timeout}s) ...")
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.get_odom() is not None or self.get_lidar() is not None:
                elapsed = time.monotonic() - start
                logger.info(
                    "DDS discovery complete — first data received in %.1fs", elapsed
                )
                _stderr(f"DDS discovery OK — data received in {elapsed:.1f}s")
                return
            time.sleep(0.1)

        # Timed out — collect diagnostic info
        import rclpy
        try:
            topics = self._node.get_topic_names_and_types()
            topic_count = len(topics)
            topic_list = [t[0] for t in topics[:15]]
        except Exception:
            topic_count = -1
            topic_list = ["(could not query)"]

        _stderr(
            f"WARNING: Warmup timed out after {timeout}s — no data received!\n"
            f"  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(not set)')}\n"
            f"  Discovered {topic_count} topics: {topic_list}\n"
            f"  odom_topic={self._odom_topic}  lidar_topic={self._lidar_topic}\n"
            f"  camera_topic={self._camera_topic}\n"
            f"  spin_thread_alive={self._spin_thread.is_alive()}"
        )

    # ------------------------------------------------------------------
    # Callbacks — cache latest messages
    # ------------------------------------------------------------------

    def _stamp_to_iso(self, stamp):
        """Convert a ROS 2 Time stamp to ISO-8601 string."""
        secs = stamp.sec + stamp.nanosec * 1e-9
        return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()

    def _camera_cb(self, msg):
        with self._image_lock:
            self._image_data = bytes(msg.data)
            self._image_stamp = self._stamp_to_iso(msg.header.stamp)

    def _lidar_cb(self, msg):
        data = {
            "ranges": list(msg.ranges),
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "timestamp": self._stamp_to_iso(msg.header.stamp),
        }
        with self._lidar_lock:
            self._lidar_data = data

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        _, _, yaw = _euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)
        data = {
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation_yaw_deg": math.degrees(yaw),
            "orientation_yaw_rad": yaw,
            "linear_velocity": {"x": lin.x, "y": lin.y},
            "angular_velocity_z": ang.z,
            "timestamp": self._stamp_to_iso(msg.header.stamp),
        }
        with self._odom_lock:
            self._odom_data = data

    # ------------------------------------------------------------------
    # Public sensor API
    # ------------------------------------------------------------------

    def get_image(self):
        """Return ``(jpeg_bytes, iso_timestamp)`` or ``(None, None)``."""
        with self._image_lock:
            return self._image_data, self._image_stamp

    def wait_for_fresh_frame(self, timeout=1.5, min_new_frames=2, poll_interval=0.05):
        """Block until a camera frame newer than the current one arrives.

        This ensures the returned image was captured *after* this method was
        called — critical for avoiding stale / motion-blurred frames after a
        turn or other motion command.

        By default this waits for **two** new frames: the first frame after a
        stop may still contain motion blur, while the second is usually stable.
        Callers can reduce ``min_new_frames`` to 1 for lower latency.

        Args:
            timeout: Maximum seconds to wait for new frames.
            min_new_frames: Number of distinct newer frames required.
            poll_interval: Poll interval in seconds while waiting.

        Returns:
            tuple: ``(jpeg_bytes, iso_timestamp)`` of the fresh frame,
                   or the latest cached frame if timeout expires.
        """
        with self._image_lock:
            old_stamp = self._image_stamp

        min_new_frames = max(1, int(min_new_frames))
        poll_interval = max(0.01, float(poll_interval))

    def drive_until_lidar_stop(self, speed=0.18, stop_distance_m=0.5,
                               max_distance_m=2.5, timeout=None):
        """Drive forward continuously and stop once front lidar reaches threshold.

        This is a fast-path approach primitive for low-clutter environments:
        no pre-check loops, no side correction strategy. It simply drives
        forward and issues an immediate stop when front distance is within
        ``stop_distance_m``.

        Args:
            speed: Forward speed in m/s.
            stop_distance_m: Stop trigger on front lidar distance (m).
            max_distance_m: Hard travel cap if trigger never occurs.
            timeout: Optional command timeout in seconds.

        Returns:
            dict with status, travelled distance and stop reason.
        """
        if timeout is None:
            timeout = DEFAULT_MOTION_TIMEOUT

        speed = min(abs(speed), MAX_LINEAR_SPEED)
        stop_distance_m = max(0.05, float(stop_distance_m))
        max_distance_m = max(0.05, float(max_distance_m))

        if not self._wait_for_odom():
            self.stop()
            return {
                "success": False,
                "message": "No odometry data available — is the robot running?",
            }

        start_odom = self.get_odom()
        sx, sy = start_odom["position"]["x"], start_odom["position"]["y"]
        start_time = time.monotonic()
        rate_sleep = 1.0 / CONTROL_HZ

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    self.stop()
                    cur = self.get_odom()
                    cx, cy = cur["position"]["x"], cur["position"]["y"]
                    dist_actual = math.hypot(cx - sx, cy - sy)
                    return {
                        "success": False,
                        "stop_reason": "timeout",
                        "distance_actual_m": round(dist_actual, 4),
                        "distance_cap_m": max_distance_m,
                        "message": f"Timed out after {timeout}s",
                    }

                front_dist = self.check_obstacle(direction="front")
                if front_dist is not None and front_dist <= stop_distance_m:
                    self.stop()
                    cur = self.get_odom()
                    cx, cy = cur["position"]["x"], cur["position"]["y"]
                    dist_actual = math.hypot(cx - sx, cy - sy)
                    return {
                        "success": True,
                        "stop_reason": "lidar_threshold",
                        "stop_distance_target_m": round(stop_distance_m, 3),
                        "front_distance_m": round(front_dist, 3),
                        "distance_actual_m": round(dist_actual, 4),
                        "start_position": {"x": round(sx, 4), "y": round(sy, 4)},
                        "end_position": {"x": round(cx, 4), "y": round(cy, 4)},
                        "message": (
                            f"Stopped on lidar trigger at {front_dist:.2f}m "
                            f"(target {stop_distance_m:.2f}m)."
                        ),
                    }

                cur = self.get_odom()
                cx, cy = cur["position"]["x"], cur["position"]["y"]
                dist_actual = math.hypot(cx - sx, cy - sy)
                if dist_actual >= max_distance_m:
                    self.stop()
                    return {
                        "success": False,
                        "stop_reason": "distance_cap",
                        "distance_actual_m": round(dist_actual, 4),
                        "distance_cap_m": max_distance_m,
                        "message": (
                            f"Reached max_distance_m ({max_distance_m:.2f}m) "
                            "before lidar stop trigger."
                        ),
                    }

                self._publish_twist(linear_x=speed)
                time.sleep(rate_sleep)
        except Exception as e:
            self.stop()
            return {
                "success": False,
                "stop_reason": "error",
                "message": f"Error during lidar-stop drive: {e}",
            }

        frames_seen = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            with self._image_lock:
                if self._image_stamp is not None and self._image_stamp != old_stamp:
                    frames_seen += 1
                    if frames_seen >= min_new_frames:
                        return self._image_data, self._image_stamp
                    # Update old_stamp to wait for the next distinct frame
                    old_stamp = self._image_stamp

        # Timeout — return whatever we have (better than nothing)
        with self._image_lock:
            return self._image_data, self._image_stamp

    # ------------------------------------------------------------------
    # Camera viewer (MJPEG stream)
    # ------------------------------------------------------------------

    def start_camera_viewer(self, port=None, open_browser=True):
        """Start a background MJPEG streaming server for the camera feed.

        Args:
            port: HTTP port for the viewer (default: 18280).
            open_browser: If True, open the viewer page in the default browser.

        Returns:
            str: URL of the viewer page.
        """
        if self._viewer_server is not None:
            url = f"http://localhost:{self._viewer_port}"
            _stderr(f"Camera viewer already running at {url}")
            if open_browser:
                if not _open_url_in_browser(url):
                    _stderr(f"Could not auto-open browser — visit {url} manually")
            return url

        port = port or DEFAULT_VIEWER_PORT

        # Create a handler class bound to this bridge instance
        handler = type('BoundMJPEGHandler', (_MJPEGHandler,), {'bridge_ref': self})

        try:
            server = _ThreadedHTTPServer(('0.0.0.0', port), handler)
        except OSError as e:
            _stderr(f"Failed to start camera viewer on port {port}: {e}")
            raise

        self._viewer_server = server
        self._viewer_port = port

        self._viewer_thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )
        self._viewer_thread.start()

        url = f"http://localhost:{port}"
        _stderr(f"Camera viewer started at {url}")

        if open_browser:
            try:
                opened = _open_url_in_browser(url)
                if not opened:
                    raise RuntimeError("No browser launcher succeeded")
            except Exception:
                _stderr(f"Could not auto-open browser — visit {url} manually")

        return url

    def stop_camera_viewer(self):
        """Stop the camera viewer HTTP server."""
        if self._viewer_server is not None:
            self._viewer_server.shutdown()
            self._viewer_server = None
            self._viewer_port = None
            _stderr("Camera viewer stopped")

    def set_mosaic(self, jpeg_bytes: bytes):
        """Store the latest panoramic mosaic for the web viewer.

        Called by rotate_and_scan after building the mosaic.
        """
        import datetime
        with self._mosaic_lock:
            self._mosaic_data = jpeg_bytes
            self._mosaic_stamp = datetime.datetime.now().isoformat(timespec='seconds')
        _stderr(f"Mosaic updated ({len(jpeg_bytes)} bytes)")

    # ------------------------------------------------------------------
    # Progressive search-frame accumulator
    # ------------------------------------------------------------------

    def add_search_frame(self, jpeg_bytes: bytes, heading_deg: float):
        """Append a camera frame captured during step-by-step search.

        Automatically rebuilds the mosaic for the web viewer so the
        operator can follow along in real time.
        """
        now = time.monotonic()
        should_rebuild = False

        with self._search_frames_lock:
            # Deduplicate near-identical heading updates to avoid expensive
            # contact-sheet rebuilds that add little information.
            if self._search_frames:
                _, last_heading = self._search_frames[-1]
                delta = abs((heading_deg - last_heading + 180.0) % 360.0 - 180.0)
                if delta < SEARCH_FRAME_MIN_HEADING_DELTA_DEG:
                    # Keep latest bytes for this heading bucket, do not append.
                    self._search_frames[-1] = (jpeg_bytes, heading_deg)
                    with self._perf_lock:
                        self._perf_stats["search_frames_skipped"] += 1
                    _stderr(
                        f"Search frame skipped (Δheading={delta:.1f}° < "
                        f"{SEARCH_FRAME_MIN_HEADING_DELTA_DEG:.1f}°)"
                    )
                else:
                    self._search_frames.append((jpeg_bytes, heading_deg))
                    with self._perf_lock:
                        self._perf_stats["search_frames_accepted"] += 1
            else:
                self._search_frames.append((jpeg_bytes, heading_deg))
                with self._perf_lock:
                    self._perf_stats["search_frames_accepted"] += 1

            # Bound memory and rebuild cost.
            if len(self._search_frames) > MAX_SEARCH_FRAMES:
                self._search_frames = self._search_frames[-MAX_SEARCH_FRAMES:]

            # Throttle rebuild frequency; mark pending updates for next slot.
            if (
                now - self._last_mosaic_rebuild_monotonic >= MOSAIC_REBUILD_MIN_INTERVAL_S
                or len(self._search_frames) <= 2
            ):
                should_rebuild = True
                self._mosaic_rebuild_pending = False
                self._last_mosaic_rebuild_monotonic = now
            else:
                self._mosaic_rebuild_pending = True

            frames = list(self._search_frames)

        _stderr(f"Search frame buffer size={len(frames)} (heading {heading_deg:.0f}°)")
        if should_rebuild:
            self._rebuild_mosaic_from_frames(frames)

    def clear_search_frames(self):
        """Reset the accumulated search frames (new search or relocation)."""
        with self._search_frames_lock:
            self._search_frames.clear()
            self._mosaic_rebuild_pending = False
            self._last_mosaic_rebuild_monotonic = 0.0
        _stderr("Search frames cleared")

    def _rebuild_mosaic_from_frames(self, frames: list[tuple[bytes, float]]):
        """Build a progressive contact-sheet grid from accumulated frames.

        Each frame is displayed as a sharp, individually labelled thumbnail
        sorted by heading — the operator sees the grid grow in the web
        viewer as the robot rotates.  No stitching is attempted, avoiding
        all seam / ghosting artefacts.
        """
        try:
            t0 = time.monotonic()
            from PIL import Image, ImageDraw, ImageFont
            import io, math

            images = []
            headings = []
            for jpeg, hdg in frames:
                img = Image.open(io.BytesIO(jpeg))
                images.append(img)
                headings.append(hdg)

            if not images:
                return

            n = len(images)

            # Heading → cardinal helper
            _DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            def _cardinal(d):
                return _DIRS[int((d % 360.0 + 22.5) / 45.0) % 8]

            # Sort by heading
            norm_hdg = [(h % 360.0) for h in headings]
            order = sorted(range(n), key=lambda i: norm_hdg[i])
            sorted_imgs = [images[i] for i in order]
            sorted_hdg = [norm_hdg[i] for i in order]

            # Thumbnail sizing
            CELL_W = 400
            first_ratio = sorted_imgs[0].height / max(sorted_imgs[0].width, 1)
            cell_h = int(CELL_W * first_ratio)
            PADDING = 6
            LABEL_H = 30

            # Grid dimensions targeting ~16:9
            best_cols = max(1, math.ceil(math.sqrt(n)))
            best_err = float('inf')
            for c in range(max(1, n // 6), n + 1):
                r = math.ceil(n / c)
                w = c * (CELL_W + PADDING) + PADDING
                h = r * (cell_h + LABEL_H + PADDING) + PADDING
                err = abs(w / max(h, 1) - 16.0 / 9.0)
                if err < best_err:
                    best_err = err
                    best_cols = c
            cols = best_cols
            rows = math.ceil(n / cols)

            sheet_w = cols * (CELL_W + PADDING) + PADDING
            sheet_h = rows * (cell_h + LABEL_H + PADDING) + PADDING
            sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
            draw = ImageDraw.Draw(sheet)

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                font = ImageFont.load_default()

            for idx in range(n):
                col = idx % cols
                row = idx // cols
                x = PADDING + col * (CELL_W + PADDING)
                y = PADDING + row * (cell_h + LABEL_H + PADDING)

                # Label bar
                draw.rectangle(
                    [x, y, x + CELL_W, y + LABEL_H], fill=(50, 50, 60))
                hdg = sorted_hdg[idx]
                orig_idx = order[idx] + 1
                label = f"[{orig_idx}]  {hdg:.0f}°  {_cardinal(hdg)}"
                draw.text((x + 8, y + 5), label,
                          fill=(255, 255, 100), font=font)

                # Thumbnail
                thumb = sorted_imgs[idx].resize(
                    (CELL_W, cell_h), Image.LANCZOS)
                sheet.paste(thumb, (x, y + LABEL_H))

                # Border
                draw.rectangle(
                    [x - 1, y - 1, x + CELL_W + 1, y + cell_h + LABEL_H + 1],
                    outline=(80, 80, 80), width=1)

            buf = io.BytesIO()
            sheet.save(buf, format="JPEG", quality=85)
            self.set_mosaic(buf.getvalue())

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            with self._perf_lock:
                count = self._perf_stats["mosaic_rebuild_count"] + 1
                prev_avg = self._perf_stats["mosaic_rebuild_avg_ms"]
                self._perf_stats["mosaic_rebuild_count"] = count
                self._perf_stats["mosaic_rebuild_last_ms"] = round(elapsed_ms, 2)
                self._perf_stats["mosaic_rebuild_avg_ms"] = round(
                    ((prev_avg * (count - 1)) + elapsed_ms) / count, 2
                )
        except Exception as exc:
            _stderr(f"WARNING: contact sheet rebuild failed: {exc}")

    def get_performance_snapshot(self):
        """Return lightweight bridge performance stats for tuning.

        Returns:
            dict: Current snapshot of mosaic/search performance counters.
        """
        with self._perf_lock:
            return dict(self._perf_stats)

    def record_tool_call(self,
                         tool_name: str,
                         reason: str,
                         args: dict | None = None,
                         arg_reasons: dict | None = None):
        entry = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "tool": tool_name,
            "reason": reason,
            "args": args or {},
            "arg_reasons": arg_reasons or {},
        }
        with self._tool_trace_lock:
            self._tool_trace.append(entry)
            if len(self._tool_trace) > self._tool_trace_max:
                self._tool_trace = self._tool_trace[-self._tool_trace_max:]

    def get_recent_tool_calls(self, limit: int = 12):
        with self._tool_trace_lock:
            return list(self._tool_trace[-max(1, limit):])

    def get_lidar_summary(self):
        scan = self.get_lidar()
        odom = self.get_odom()
        if scan is None:
            return {
                "status": "no-data",
                "message": "No LiDAR data available yet",
            }

        ranges = scan.get("ranges") or []
        total = len(ranges)
        if total == 0:
            return {
                "status": "no-data",
                "message": "Empty LiDAR scan",
            }

        angle_min = float(scan.get("angle_min", 0.0))
        angle_increment = float(scan.get("angle_increment", 0.0))

        lidar_rotation_rad = math.radians(LIDAR_FRAME_ROTATION_DEG)

        def _beam_angle_deg(index: int) -> float:
            beam_rad = angle_min + index * angle_increment
            deg = (math.degrees(beam_rad) + LIDAR_FRAME_ROTATION_DEG) % 360.0
            return deg

        def _slice_stats(center_deg: float, half_width_deg: float = 30.0):
            valid = []
            for idx, value in enumerate(ranges):
                if value <= 0 or math.isinf(value):
                    continue
                angle = _beam_angle_deg(idx)
                diff = abs(angle - center_deg)
                if diff > 180.0:
                    diff = 360.0 - diff
                if diff <= half_width_deg:
                    valid.append(value)
            if not valid:
                return {"min_m": None, "avg_m": None}
            return {
                "min_m": round(min(valid), 3),
                "avg_m": round(sum(valid) / len(valid), 3),
            }

        sectors = {
            "front": _slice_stats(0.0),
            "left": _slice_stats(90.0),
            "back": _slice_stats(180.0),
            "right": _slice_stats(270.0),
        }

        sampled_points = []
        if odom is not None:
            robot_x = float(odom["position"]["x"])
            robot_y = float(odom["position"]["y"])
            robot_heading_rad = float(odom["orientation_yaw_rad"])

            step = max(1, total // 120)
            for idx in range(0, total, step):
                distance = ranges[idx]
                if distance <= 0 or math.isinf(distance):
                    continue

                local_angle_rad = angle_min + idx * angle_increment + lidar_rotation_rad
                world_angle_rad = robot_heading_rad + local_angle_rad
                world_x = robot_x + distance * math.cos(world_angle_rad)
                world_y = robot_y + distance * math.sin(world_angle_rad)

                sampled_points.append({
                    "distance_m": round(float(distance), 3),
                    "angle_deg": round(_beam_angle_deg(idx), 1),
                    "world_x_m": round(world_x, 3),
                    "world_y_m": round(world_y, 3),
                })

        return {
            "status": "ok",
            "timestamp": scan.get("timestamp"),
            "range_limits_m": {
                "min": scan.get("range_min"),
                "max": scan.get("range_max"),
            },
            "sectors": sectors,
            "points": sampled_points,
        }

    def get_operator_status(self):
        odom = self.get_odom()
        perf = self.get_performance_snapshot()
        lidar = self.get_lidar_summary()
        recent_calls = self.get_recent_tool_calls(limit=10)

        overview = "Idle"
        if recent_calls:
            latest = recent_calls[-1]
            overview = f"Last action: {latest['tool']} — {latest['reason']}"
        if odom and odom.get("linear_velocity", {}).get("x", 0.0) > 0.02:
            overview = "Robot moving while monitoring camera/lidar"

        pose = None
        if odom:
            pose = {
                "x_m": round(odom["position"]["x"], 3),
                "y_m": round(odom["position"]["y"], 3),
                "heading_deg": round(odom["orientation_yaw_deg"], 1),
                "linear_m_s": round(odom["linear_velocity"]["x"], 3),
                "angular_rad_s": round(odom["angular_velocity_z"], 3),
            }

        return {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "overview": overview,
            "pose": pose,
            "lidar": lidar,
            "recent_tools": recent_calls,
            "performance": perf,
        }

    @property
    def camera_viewer_url(self):
        """Return the viewer URL if running, else None."""
        if self._viewer_port is not None:
            return f"http://localhost:{self._viewer_port}"
        return None

    def get_lidar(self):
        """Return latest lidar scan dict or ``None``."""
        with self._lidar_lock:
            return self._lidar_data

    def get_odom(self):
        """Return latest odometry dict or ``None``."""
        with self._odom_lock:
            return self._odom_data

    # ------------------------------------------------------------------
    # Lidar obstacle detection helpers
    # ------------------------------------------------------------------

    def check_obstacle(self, direction="front", arc_half_angle_deg=30):
        """Return the minimum range (m) in a given direction arc.

        Args:
            direction: One of 'front', 'left', 'right', 'back'.
            arc_half_angle_deg: Half-width of the arc to check.

        Returns:
            float or None: Minimum valid range in the arc, or None if no data.
        """
        scan = self.get_lidar()
        if scan is None:
            return None

        ranges = scan["ranges"]
        n = len(ranges)
        if n == 0:
            return None

        # Centre angle for each direction (LDS-02: 0° = front, CCW)
        centres = {"front": 0, "left": 90, "back": 180, "right": 270}
        centre = centres.get(direction, 0)

        min_range = float("inf")
        for i, r in enumerate(ranges):
            angle = (i / n) * 360.0
            # Angular distance from centre (handles wrap-around)
            diff = abs(angle - centre)
            if diff > 180:
                diff = 360 - diff
            if diff <= arc_half_angle_deg:
                if not math.isinf(r) and r > 0:
                    min_range = min(min_range, r)

        return min_range if not math.isinf(min_range) else None

    def check_path_clear(self, direction="front", threshold_m=DEFAULT_SAFETY_MARGIN_M):
        """Check whether a direction is clear of obstacles.

        Returns:
            dict with {clear: bool, min_distance_m: float|None, direction: str, threshold_m: float}
        """
        min_dist = self.check_obstacle(direction=direction)
        if min_dist is None:
            return {
                "clear": False,
                "min_distance_m": None,
                "direction": direction,
                "threshold_m": threshold_m,
                "reason": "No lidar data available",
            }
        return {
            "clear": min_dist >= threshold_m,
            "min_distance_m": round(min_dist, 3),
            "direction": direction,
            "threshold_m": threshold_m,
        }

    # ------------------------------------------------------------------
    # Trapezoidal velocity profile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trapezoidal_speed(dist_done, dist_total, target_speed,
                           ramp_dist=LINEAR_RAMP_DISTANCE,
                           min_speed=MIN_LINEAR_SPEED):
        """Compute linear speed for a trapezoidal velocity profile.

        Ramps up over the first ``ramp_dist`` metres, holds at ``target_speed``,
        then ramps down over the last ``ramp_dist`` metres.
        """
        dist_remaining = dist_total - dist_done

        # Short moves where ramp-up + ramp-down > total distance
        effective_ramp = min(ramp_dist, dist_total / 2.0)
        if effective_ramp < 0.001:
            return target_speed

        if dist_done < effective_ramp:
            # Ramp up
            frac = dist_done / effective_ramp
            speed = min_speed + frac * (target_speed - min_speed)
        elif dist_remaining < effective_ramp:
            # Ramp down
            frac = dist_remaining / effective_ramp
            speed = min_speed + frac * (target_speed - min_speed)
        else:
            # Cruise
            speed = target_speed

        return max(min_speed, min(speed, target_speed))

    @staticmethod
    def _trapezoidal_angular_speed(angle_done_rad, angle_total_rad, target_speed,
                                   ramp_rad=math.radians(ANGULAR_RAMP_DEG),
                                   min_speed=MIN_ANGULAR_SPEED):
        """Compute angular speed for a trapezoidal velocity profile."""
        angle_remaining = angle_total_rad - angle_done_rad

        effective_ramp = min(ramp_rad, angle_total_rad / 2.0)
        if effective_ramp < 0.001:
            return target_speed

        if angle_done_rad < effective_ramp:
            frac = angle_done_rad / effective_ramp
            speed = min_speed + frac * (target_speed - min_speed)
        elif angle_remaining < effective_ramp:
            frac = angle_remaining / effective_ramp
            speed = min_speed + frac * (target_speed - min_speed)
        else:
            speed = target_speed

        return max(min_speed, min(speed, target_speed))

    @staticmethod
    def _smooth_linear_speed(previous_speed, target_speed, dt,
                             alpha=LINEAR_CMD_SMOOTHING_ALPHA,
                             accel_limit=LINEAR_ACCEL_LIMIT,
                             decel_limit=LINEAR_DECEL_LIMIT):
        """Smooth linear velocity command to reduce jitter.

        Applies low-pass filtering plus accel/decel rate limits so
        published cmd_vel changes are gradual and stable.
        """
        dt = max(1e-3, float(dt))
        alpha = max(0.0, min(1.0, float(alpha)))

        target_speed = max(0.0, float(target_speed))
        previous_speed = max(0.0, float(previous_speed))

        filtered = previous_speed + alpha * (target_speed - previous_speed)

        if filtered > previous_speed:
            max_delta = max(0.0, float(accel_limit)) * dt
            return min(filtered, previous_speed + max_delta)

        max_delta = max(0.0, float(decel_limit)) * dt
        return max(filtered, previous_speed - max_delta)

    # ------------------------------------------------------------------
    # Public motion API
    # ------------------------------------------------------------------

    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        """Publish a Twist message on /cmd_vel."""
        twist = self._Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(twist)

    def stop(self):
        """Emergency stop — publish zero velocity multiple times for reliable braking."""
        for _ in range(5):
            self._publish_twist(0.0, 0.0)
            time.sleep(0.02)
        logger.info("STOP command sent")

    def _wait_for_odom(self, timeout=5.0):
        """Block until at least one odometry message has been received."""
        start = time.monotonic()
        while self.get_odom() is None:
            if time.monotonic() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    def move_forward(self, distance_m=0.5, speed=0.2, timeout=None,
                     safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
        """Move forward ``distance_m`` metres at ``speed`` m/s.

        Uses trapezoidal velocity ramping for smooth start/stop and
        reactive lidar obstacle checking on every control tick.

        Returns a dict with ``{success, distance_requested, distance_actual,
        start_position, end_position, message}``.
        """
        if timeout is None:
            timeout = DEFAULT_MOTION_TIMEOUT

        # Clamp speed to Burger limits
        speed = min(abs(speed), MAX_LINEAR_SPEED)
        distance_m = abs(distance_m)

        if not self._wait_for_odom():
            self.stop()
            return {
                "success": False,
                "message": "No odometry data available — is the robot running?",
            }

        start_odom = self.get_odom()
        sx, sy = start_odom["position"]["x"], start_odom["position"]["y"]

        start_time = time.monotonic()
        rate_sleep = 1.0 / CONTROL_HZ
        prev_cmd_speed = 0.0

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    self.stop()
                    cur = self.get_odom()
                    cx, cy = cur["position"]["x"], cur["position"]["y"]
                    dist_actual = math.hypot(cx - sx, cy - sy)
                    return {
                        "success": False,
                        "distance_requested_m": distance_m,
                        "distance_actual_m": round(dist_actual, 4),
                        "start_position": {"x": sx, "y": sy},
                        "end_position": {"x": cx, "y": cy},
                        "message": f"Timed out after {timeout}s",
                    }

                # Reactive lidar obstacle check
                front_dist = self.check_obstacle(direction="front")
                if front_dist is not None and front_dist < safety_margin_m:
                    self.stop()
                    cur = self.get_odom()
                    cx, cy = cur["position"]["x"], cur["position"]["y"]
                    dist_actual = math.hypot(cx - sx, cy - sy)
                    return {
                        "success": False,
                        "obstacle_detected": True,
                        "obstacle_distance_m": round(front_dist, 3),
                        "distance_requested_m": distance_m,
                        "distance_actual_m": round(dist_actual, 4),
                        "start_position": {"x": round(sx, 4), "y": round(sy, 4)},
                        "end_position": {"x": round(cx, 4), "y": round(cy, 4)},
                        "message": (
                            f"Obstacle detected {front_dist:.2f}m ahead "
                            f"(safety margin {safety_margin_m}m). Stopped after "
                            f"{dist_actual:.3f}m of {distance_m}m."
                        ),
                    }

                cur = self.get_odom()
                cx, cy = cur["position"]["x"], cur["position"]["y"]
                dist_actual = math.hypot(cx - sx, cy - sy)

                if dist_actual >= distance_m:
                    self.stop()
                    return {
                        "success": True,
                        "distance_requested_m": distance_m,
                        "distance_actual_m": round(dist_actual, 4),
                        "start_position": {"x": round(sx, 4), "y": round(sy, 4)},
                        "end_position": {"x": round(cx, 4), "y": round(cy, 4)},
                        "message": "Target distance reached",
                    }

                # Trapezoidal velocity profile
                target_cmd_speed = self._trapezoidal_speed(
                    dist_actual, distance_m, speed
                )
                cmd_speed = self._smooth_linear_speed(
                    previous_speed=prev_cmd_speed,
                    target_speed=target_cmd_speed,
                    dt=rate_sleep,
                )
                prev_cmd_speed = cmd_speed

                self._publish_twist(linear_x=cmd_speed)
                time.sleep(rate_sleep)
        except Exception as e:
            self.stop()
            return {"success": False, "message": f"Error during move: {e}"}

    def turn(self, angle_deg=90.0, angular_speed=0.25, timeout=None):
        """Turn ``angle_deg`` degrees (positive = CCW, negative = CW).

        Uses trapezoidal angular velocity ramping for smooth rotation and
        checks front lidar to avoid rotating into immediate obstacles.

        Returns a dict with ``{success, angle_requested_deg, angle_actual_deg,
        start_yaw_deg, end_yaw_deg, message}``.
        """
        if timeout is None:
            timeout = DEFAULT_MOTION_TIMEOUT

        # Clamp angular speed
        angular_speed = min(abs(angular_speed), MAX_ANGULAR_SPEED)
        target_rad = math.radians(angle_deg)
        direction = 1.0 if target_rad >= 0 else -1.0
        target_rad_abs = abs(target_rad)

        if not self._wait_for_odom():
            self.stop()
            return {
                "success": False,
                "message": "No odometry data available — is the robot running?",
            }

        start_odom = self.get_odom()
        start_yaw = start_odom["orientation_yaw_rad"]
        accumulated = 0.0
        prev_yaw = start_yaw

        start_time = time.monotonic()
        rate_sleep = 1.0 / CONTROL_HZ

        try:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    self.stop()
                    cur = self.get_odom()
                    return {
                        "success": False,
                        "angle_requested_deg": angle_deg,
                        "angle_actual_deg": round(math.degrees(accumulated), 2),
                        "start_yaw_deg": round(math.degrees(start_yaw), 2),
                        "end_yaw_deg": round(cur["orientation_yaw_deg"], 2),
                        "message": f"Timed out after {timeout}s",
                    }

                cur = self.get_odom()
                cur_yaw = cur["orientation_yaw_rad"]
                delta = _normalize_angle(cur_yaw - prev_yaw)
                accumulated += delta
                prev_yaw = cur_yaw

                if abs(accumulated) >= target_rad_abs:
                    self.stop()
                    # Wait for the robot to physically settle, then read
                    # final yaw from odometry for an accurate report.
                    time.sleep(0.1)
                    final = self.get_odom()
                    final_yaw = final["orientation_yaw_rad"] if final else cur_yaw
                    final_acc = accumulated + _normalize_angle(final_yaw - prev_yaw)
                    return {
                        "success": True,
                        "angle_requested_deg": angle_deg,
                        "angle_actual_deg": round(math.degrees(final_acc), 2),
                        "start_yaw_deg": round(math.degrees(start_yaw), 2),
                        "end_yaw_deg": round(math.degrees(final_yaw), 2),
                        "message": "Target angle reached",
                    }

                # Trapezoidal angular velocity profile
                cmd_angular = self._trapezoidal_angular_speed(
                    abs(accumulated), target_rad_abs, angular_speed
                )
                self._publish_twist(angular_z=direction * cmd_angular)
                time.sleep(rate_sleep)
        except Exception as e:
            self.stop()
            return {"success": False, "message": f"Error during turn: {e}"}

    def turn_to_heading(self, target_deg, tolerance_deg=3.0,
                        max_corrections=2, angular_speed=0.5):
        """Turn to face an absolute heading (in degrees).

        Computes the shortest rotation path (handling 0/360 wrap-around)
        and optionally performs corrective micro-turns if the residual error
        exceeds *tolerance_deg*.

        Args:
            target_deg: Desired heading in degrees (0–360).
            tolerance_deg: Acceptable error in degrees (default 3.0).
            max_corrections: Maximum corrective micro-turns (default 2).
            angular_speed: Angular speed in rad/s (default 0.5).

        Returns:
            dict with ``{success, target_heading_deg, actual_heading_deg,
            error_deg, corrections_made, message}``.
        """
        if not self._wait_for_odom():
            return {
                "success": False,
                "target_heading_deg": target_deg,
                "message": "No odometry data available — is the robot running?",
            }

        corrections_made = 0

        for attempt in range(1 + max_corrections):
            odom = self.get_odom()
            current_deg = odom["orientation_yaw_deg"]

            # Shortest path: result in [-180, 180]
            delta = (target_deg - current_deg + 180.0) % 360.0 - 180.0

            if abs(delta) <= tolerance_deg:
                return {
                    "success": True,
                    "target_heading_deg": round(target_deg, 2),
                    "actual_heading_deg": round(current_deg, 2),
                    "error_deg": round(abs(delta), 2),
                    "corrections_made": corrections_made,
                    "message": (
                        "On target" if attempt == 0
                        else f"On target after {corrections_made} correction(s)"
                    ),
                }

            # Execute the turn
            result = self.turn(angle_deg=delta, angular_speed=angular_speed)

            if attempt > 0:
                corrections_made += 1

            if not result.get("success"):
                # Turn failed — report with current state
                final_odom = self.get_odom()
                final_deg = final_odom["orientation_yaw_deg"] if final_odom else current_deg
                final_error = abs((target_deg - final_deg + 180.0) % 360.0 - 180.0)
                return {
                    "success": False,
                    "target_heading_deg": round(target_deg, 2),
                    "actual_heading_deg": round(final_deg, 2),
                    "error_deg": round(final_error, 2),
                    "corrections_made": corrections_made,
                    "message": f"Turn failed: {result.get('message', '?')}",
                }

        # All attempts exhausted — return best effort
        final_odom = self.get_odom()
        final_deg = final_odom["orientation_yaw_deg"] if final_odom else target_deg
        final_error = abs((target_deg - final_deg + 180.0) % 360.0 - 180.0)
        return {
            "success": final_error <= tolerance_deg,
            "target_heading_deg": round(target_deg, 2),
            "actual_heading_deg": round(final_deg, 2),
            "error_deg": round(final_error, 2),
            "corrections_made": corrections_made,
            "message": (
                "On target" if final_error <= tolerance_deg
                else f"Residual error {final_error:.1f}° after {corrections_made} correction(s)"
            ),
        }

    def navigate_safely(self, distance_m=0.5, speed=0.15,
                        obstacle_threshold_m=0.3):
        """Navigate forward with integrated lidar-based obstacle avoidance.

        If the front is blocked, attempts a corrective turn toward the clearer
        side (left or right), then retries forward motion. Makes up to 3
        correction attempts before giving up.

        Returns a dict with navigation result.
        """
        speed = min(abs(speed), MAX_LINEAR_SPEED)
        distance_m = abs(distance_m)
        max_corrections = 3

        if not self._wait_for_odom():
            self.stop()
            return {"success": False, "message": "No odometry data available."}

        start_odom = self.get_odom()
        sx, sy = start_odom["position"]["x"], start_odom["position"]["y"]
        total_moved = 0.0
        corrections = 0

        while total_moved < distance_m:
            remaining = distance_m - total_moved

            # Check front clearance
            front_dist = self.check_obstacle(direction="front")
            if front_dist is not None and front_dist < obstacle_threshold_m:
                if corrections >= max_corrections:
                    self.stop()
                    cur = self.get_odom()
                    return {
                        "success": False,
                        "distance_requested_m": distance_m,
                        "distance_actual_m": round(total_moved, 4),
                        "corrections_attempted": corrections,
                        "message": (
                            f"Path blocked after {corrections} correction attempts. "
                            f"Front obstacle at {front_dist:.2f}m. Moved {total_moved:.3f}m "
                            f"of {distance_m}m."
                        ),
                    }

                # Determine which side is clearer
                left_dist = self.check_obstacle(direction="left") or 0
                right_dist = self.check_obstacle(direction="right") or 0

                if left_dist >= right_dist:
                    turn_angle = 30.0   # Turn left
                else:
                    turn_angle = -30.0  # Turn right

                _stderr(f"Obstacle at {front_dist:.2f}m — correcting {turn_angle}°")
                self.turn(angle_deg=turn_angle, angular_speed=0.4)
                corrections += 1
                continue

            # Move a segment (max 0.3m per segment for frequent checking)
            segment = min(remaining, 0.3)
            result = self.move_forward(
                distance_m=segment, speed=speed,
                safety_margin_m=obstacle_threshold_m,
            )

            if result.get("success"):
                total_moved += result.get("distance_actual_m", 0)
            elif result.get("obstacle_detected"):
                total_moved += result.get("distance_actual_m", 0)
                # Let the loop try a correction on next iteration
                corrections += 1
                if corrections > max_corrections:
                    cur = self.get_odom()
                    return {
                        "success": False,
                        "distance_requested_m": distance_m,
                        "distance_actual_m": round(total_moved, 4),
                        "corrections_attempted": corrections,
                        "message": result["message"],
                    }
                continue
            else:
                # Other failure (timeout, no odom, etc.)
                return result

        cur = self.get_odom()
        cx, cy = cur["position"]["x"], cur["position"]["y"]
        return {
            "success": True,
            "distance_requested_m": distance_m,
            "distance_actual_m": round(math.hypot(cx - sx, cy - sy), 4),
            "corrections_attempted": corrections,
            "start_position": {"x": round(sx, 4), "y": round(sy, 4)},
            "end_position": {"x": round(cx, 4), "y": round(cy, 4)},
            "message": (
                f"Navigated {distance_m}m with {corrections} obstacle corrections."
                if corrections > 0
                else f"Navigated {distance_m}m — path was clear."
            ),
        }

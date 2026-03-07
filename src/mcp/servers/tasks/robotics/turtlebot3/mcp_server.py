"""
TurtleBot3 MCP Server — sensor and motion tools for a TurtleBot3 robot via ROS 2.

10 Core Tools:
  Sensors:
    1. get_camera_image    — Capture latest camera frame (returns ImageContent for VLMs)
    2. get_lidar_scan      — Latest 360° LiDAR scan (ranges, angles, limits)
    3. get_odometry        — Current pose & velocity (dead reckoning)
    4. check_path_clear    — Quick lidar check if a direction is obstacle-free
  Motion:
    5. move_forward        — Drive forward N metres (lidar-guarded, smooth ramping)
    6. turn                — Rotate N degrees (smooth ramping)
    7. navigate_safely     — Move forward with automatic obstacle avoidance
    8. stop                — Emergency stop
  Utility:
    9. open_camera_viewer  — Open live MJPEG camera stream in browser
   10. diagnose_ros        — ROS 2 connectivity diagnostics

Designed for TurtleBot3 Burger on ROS 2 Humble.
"""

import base64
import json
import os
import sys
import logging

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _stderr(msg: str):
    """Print diagnostic to stderr so it isn't swallowed by multiprocessing."""
    print(f"[TurtleBot3] {msg}", file=sys.stderr, flush=True)

mcp = FastMCP("TurtleBot3 MCP Server")

# Lazy singleton reference — initialised in run()
_bridge = None


def _get_bridge():
    """Return the TurtleBot3Bridge singleton (must be initialised via run())."""
    global _bridge
    if _bridge is None:
        raise RuntimeError(
            "TurtleBot3Bridge not initialised. Start this server via run()."
        )
    return _bridge


# =========================================================================
# SENSOR TOOLS
# =========================================================================

@mcp.tool(
    title="Get Camera Image",
    description=(
        "Capture a FRESH camera frame from the TurtleBot3. "
        "Waits for a new frame to arrive (up to 1 s) so the image is never "
        "stale or motion-blurred from a recent turn. "
        "Returns a JPEG image that can be analysed by a vision-language model. "
        "Use this tool when asked to describe surroundings or identify objects."
    ),
)
def get_camera_image() -> Image:
    """Return a fresh compressed camera image as ImageContent."""
    bridge = _get_bridge()
    # wait_for_fresh_frame blocks until a frame newer than the current
    # cached one arrives, avoiding stale / motion-blurred images.
    image_bytes, timestamp = bridge.wait_for_fresh_frame(timeout=1.0)

    if image_bytes is None:
        raise ValueError(
            "No camera image available yet. The camera may not be publishing, "
            "or no frame has been received. Check that the camera node is running "
            f"on topic '{bridge._camera_topic}'."
        )

    # Feed frame into progressive mosaic for the web viewer
    try:
        odom = bridge.get_odom()
        heading = odom["yaw_deg"] if odom else 0.0
        bridge.add_search_frame(image_bytes, heading)
    except Exception:
        pass  # non-critical — don't break the camera tool

    # FastMCP Image() accepts raw bytes and returns ImageContent
    return Image(data=image_bytes, format="jpeg")


@mcp.tool(
    title="Get LiDAR Scan",
    description=(
        "Get the latest 360° LiDAR scan from the TurtleBot3. "
        "Returns an array of distance measurements (in metres) around the robot, "
        "along with angle information. Use this to detect obstacles before moving. "
        "Ranges of 'inf' or 0.0 indicate no detection in that direction."
    ),
)
def get_lidar_scan(
    summarize: bool = True,
) -> str:
    """Return latest LiDAR scan data as JSON.

    Args:
        summarize: If True (default), return a compact summary with min/max/avg
            per quadrant instead of the full 360-sample array.
    """
    bridge = _get_bridge()
    scan = bridge.get_lidar()

    if scan is None:
        return json.dumps({
            "error": (
                "No LiDAR data available yet. Check that the LiDAR node is "
                f"running on topic '{bridge._lidar_topic}'."
            )
        })

    if not summarize:
        return json.dumps(scan, default=_json_default)

    # Compact summary: split into 4 quadrants (front, left, back, right)
    ranges = scan["ranges"]
    n = len(ranges)
    if n == 0:
        return json.dumps({"error": "Empty LiDAR scan"})

    quadrants = {
        "front": _range_stats(ranges, 0, n, -45, 45),
        "left": _range_stats(ranges, 0, n, 45, 135),
        "back": _range_stats(ranges, 0, n, 135, 225),
        "right": _range_stats(ranges, 0, n, 225, 315),
    }

    return json.dumps({
        "quadrants": quadrants,
        "total_points": n,
        "range_min_m": scan["range_min"],
        "range_max_m": scan["range_max"],
        "timestamp": scan["timestamp"],
    }, indent=2, default=_json_default)


@mcp.tool(
    title="Get Odometry",
    description=(
        "Get the current odometry (pose and velocity) of the TurtleBot3. "
        "Returns position (x, y in metres), heading (yaw in degrees), "
        "and current velocities. Use this for dead reckoning and to confirm "
        "the robot's position before and after motion commands."
    ),
)
def get_odometry() -> str:
    """Return current odometry as JSON."""
    bridge = _get_bridge()
    odom = bridge.get_odom()

    if odom is None:
        return json.dumps({
            "error": (
                "No odometry data available yet. Check that the robot base is "
                f"running and publishing on topic '{bridge._odom_topic}'."
            )
        })

    # Return a clean subset (drop raw radians)
    return json.dumps({
        "position_m": {
            "x": round(odom["position"]["x"], 4),
            "y": round(odom["position"]["y"], 4),
        },
        "heading_deg": round(odom["orientation_yaw_deg"], 2),
        "linear_velocity_m_s": round(odom["linear_velocity"]["x"], 4),
        "angular_velocity_deg_s": round(
            odom["angular_velocity_z"] * 57.2958, 2  # rad/s → deg/s
        ),
        "timestamp": odom["timestamp"],
    }, indent=2)


# =========================================================================
# MOTION TOOLS
# =========================================================================

@mcp.tool(
    title="Move Forward",
    description=(
        "Move the TurtleBot3 forward by a specified distance in metres. "
        "Default speed is 0.2 m/s (max 0.22 m/s for the Burger model). "
        "The robot will stop automatically when the target distance is reached "
        "or after a safety timeout. Always check LiDAR for obstacles first!"
    ),
)
def move_forward(
    distance: float = 0.5,
    speed: float = 0.2,
) -> str:
    """Move the robot forward.

    Args:
        distance: Distance to travel in metres (positive). Default 0.5 m.
        speed: Linear speed in m/s (0 < speed ≤ 0.22). Default 0.2 m/s.
    """
    if distance <= 0:
        return json.dumps({"error": "Distance must be positive.", "status": "failed"})
    if speed <= 0:
        return json.dumps({"error": "Speed must be positive.", "status": "failed"})

    bridge = _get_bridge()
    result = bridge.move_forward(distance_m=distance, speed=speed)
    return json.dumps(result, indent=2, default=_json_default)


@mcp.tool(
    title="Turn",
    description=(
        "Turn the TurtleBot3 by a specified angle in degrees. "
        "Positive angle = counter-clockwise (left), negative = clockwise (right). "
        "Default angular speed is 0.5 rad/s (max 2.84 rad/s). "
        "The robot stops automatically when the target angle is reached."
    ),
)
def turn(
    angle: float = 90.0,
    speed: float = 0.5,
) -> str:
    """Turn the robot.

    Args:
        angle: Angle to turn in degrees. Positive = CCW/left, negative = CW/right.
        speed: Angular speed in rad/s (0 < speed ≤ 2.84). Default 0.5 rad/s.
    """
    if speed <= 0:
        return json.dumps({"error": "Speed must be positive.", "status": "failed"})
    if angle == 0:
        return json.dumps({"error": "Angle must be non-zero.", "status": "failed"})

    bridge = _get_bridge()
    result = bridge.turn(angle_deg=angle, angular_speed=speed)
    return json.dumps(result, indent=2, default=_json_default)


@mcp.tool(
    title="Stop",
    description=(
        "Immediately stop the TurtleBot3 by publishing zero velocity. "
        "Use this as an emergency stop or when you want to halt all motion."
    ),
)
def stop() -> str:
    """Send an immediate stop command."""
    bridge = _get_bridge()
    bridge.stop()
    odom = bridge.get_odom()
    pos = odom or {}
    return json.dumps({
        "status": "stopped",
        "current_position": pos.get("position"),
        "current_heading_deg": pos.get("orientation_yaw_deg"),
        "message": "Zero velocity published — robot stopped.",
    }, indent=2, default=_json_default)


# =========================================================================
# LIDAR NAVIGATION TOOLS
# =========================================================================

@mcp.tool(
    title="Check Path Clear",
    description=(
        "Quick lidar check whether a direction is free of obstacles. "
        "Returns whether the path is clear and the minimum distance to the "
        "nearest object. Use this before moving to verify safety, or to "
        "decide which direction to turn when the front path is blocked."
    ),
)
def check_path_clear(
    direction: str = "front",
    threshold_m: float = 0.3,
) -> str:
    """Check if a direction is obstacle-free.

    Args:
        direction: Direction to check — 'front', 'left', 'right', or 'back'.
        threshold_m: Distance threshold in metres. If closest obstacle is
            nearer than this, the path is not clear. Default 0.3 m.
    """
    if direction not in ("front", "left", "right", "back"):
        return json.dumps({
            "error": f"Invalid direction '{direction}'. Use front/left/right/back.",
        })

    bridge = _get_bridge()
    result = bridge.check_path_clear(direction=direction, threshold_m=threshold_m)
    return json.dumps(result, indent=2, default=_json_default)


@mcp.tool(
    title="Navigate Safely",
    description=(
        "Move the TurtleBot3 forward with integrated lidar obstacle avoidance. "
        "Unlike plain move_forward, this tool automatically checks the path "
        "and attempts corrective turns if an obstacle is detected. Use this "
        "when navigating through cluttered environments (desks, chairs, cables). "
        "Default speed is 0.15 m/s (slightly slower for safety). "
        "Returns details about the path taken, including any corrections."
    ),
)
def navigate_safely(
    distance: float = 0.5,
    speed: float = 0.15,
    obstacle_threshold_m: float = 0.3,
) -> str:
    """Navigate forward with automatic obstacle avoidance.

    Args:
        distance: Distance to travel in metres (positive). Default 0.5 m.
        speed: Linear speed in m/s (0 < speed ≤ 0.22). Default 0.15 m/s.
        obstacle_threshold_m: Stop/correct if obstacle closer than this. Default 0.3 m.
    """
    if distance <= 0:
        return json.dumps({"error": "Distance must be positive.", "status": "failed"})
    if speed <= 0:
        return json.dumps({"error": "Speed must be positive.", "status": "failed"})

    bridge = _get_bridge()
    result = bridge.navigate_safely(
        distance_m=distance,
        speed=speed,
        obstacle_threshold_m=obstacle_threshold_m,
    )
    return json.dumps(result, indent=2, default=_json_default)


# =========================================================================
# CAMERA VIEWER TOOL
# =========================================================================

@mcp.tool(
    title="Open Camera Viewer",
    description=(
        "Open a live camera viewer in the operator's web browser. "
        "The viewer shows an MJPEG stream of the TurtleBot3 camera in real-time. "
        "Use this before or during vision tasks so the operator can see what "
        "the robot sees. Returns the URL of the viewer page."
    ),
)
def open_camera_viewer(
    port: int = 18280,
) -> str:
    """Start the MJPEG camera viewer and open it in a browser.

    Args:
        port: HTTP port for the viewer (default: 18280).
    """
    bridge = _get_bridge()

    try:
        url = bridge.start_camera_viewer(port=port, open_browser=True)
        return json.dumps({
            "status": "running",
            "url": url,
            "message": (
                f"Camera viewer is live at {url}. "
                "The browser should open automatically. "
                "The stream updates at ~15 fps when camera data is available."
            ),
        }, indent=2)
    except OSError as e:
        return json.dumps({
            "status": "error",
            "message": f"Could not start viewer: {e}",
        }, indent=2)


# =========================================================================
# DIAGNOSTIC TOOL
# =========================================================================

@mcp.tool(
    title="Diagnose ROS",
    description=(
        "Run ROS 2 diagnostics inside the MCP server process. "
        "Reports environment variables, DDS settings, node graph info, "
        "subscription status, and whether any data has been received. "
        "Use this to debug connection problems."
    ),
)
def diagnose_ros() -> str:
    """Return detailed ROS 2 diagnostic information."""
    diag = {}

    # Environment
    diag["environment"] = {
        "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "(not set)"),
        "RMW_IMPLEMENTATION": os.environ.get("RMW_IMPLEMENTATION", "(default)"),
        "ROS_LOCALHOST_ONLY": os.environ.get("ROS_LOCALHOST_ONLY", "(not set)"),
        "CYCLONEDDS_URI": os.environ.get("CYCLONEDDS_URI", "(not set)"),
        "FASTRTPS_DEFAULT_PROFILES_FILE": os.environ.get(
            "FASTRTPS_DEFAULT_PROFILES_FILE", "(not set)"
        ),
        "LD_LIBRARY_PATH_has_ros": "/opt/ros" in os.environ.get("LD_LIBRARY_PATH", ""),
        "AMENT_PREFIX_PATH": os.environ.get("AMENT_PREFIX_PATH", "(not set)"),
        "PID": os.getpid(),
    }

    # Bridge state
    try:
        bridge = _get_bridge()
        diag["bridge"] = {
            "camera_topic": bridge._camera_topic,
            "lidar_topic": bridge._lidar_topic,
            "odom_topic": bridge._odom_topic,
            "cmd_vel_topic": bridge._cmd_vel_topic,
            "has_image": bridge._image_data is not None,
            "has_lidar": bridge._lidar_data is not None,
            "has_odom": bridge._odom_data is not None,
            "image_stamp": bridge._image_stamp,
            "spin_thread_alive": bridge._spin_thread.is_alive() if bridge._spin_thread else False,
            "shutdown_flag": bridge._shutdown,
        }

        # Try to get node graph info from within rclpy
        try:
            node = bridge._node
            topic_names_and_types = node.get_topic_names_and_types()
            diag["ros_graph"] = {
                "node_name": node.get_name(),
                "node_namespace": node.get_namespace(),
                "discovered_topics": [
                    {"name": name, "types": types}
                    for name, types in topic_names_and_types
                ],
                "discovered_topic_count": len(topic_names_and_types),
            }

            # Check subscription counts for our topics
            diag["subscription_info"] = {}
            for topic in [bridge._camera_topic, bridge._lidar_topic,
                          bridge._odom_topic, bridge._cmd_vel_topic]:
                try:
                    pubs = node.get_publishers_info_by_topic(topic)
                    diag["subscription_info"][topic] = {
                        "publishers_found": len(pubs),
                        "publishers": [
                            {"node": p.node_name, "qos_reliability": str(p.qos_profile.reliability)}
                            for p in pubs
                        ] if pubs else [],
                    }
                except Exception as e:
                    diag["subscription_info"][topic] = {"error": str(e)}

        except Exception as e:
            diag["ros_graph"] = {"error": str(e)}

    except RuntimeError as e:
        diag["bridge"] = {"error": str(e)}

    return json.dumps(diag, indent=2, default=_json_default)


# =========================================================================
# PANORAMIC STITCHING / MOSAIC BUILDERS
# =========================================================================

# Camera horizontal field-of-view in degrees (TurtleBot3 Burger Raspberry Pi
# Camera Module v2 — measured/estimated).
_CAMERA_HFOV_DEG = 62.0


def _build_grid_mosaic(
    jpeg_frames: list[bytes],
    headings: list[float],
    cell_width: int = 320,
) -> bytes:
    """Arrange captured frames into a labeled grid image (fallback layout).

    Returns JPEG bytes of the mosaic.  Each cell is resized to *cell_width*
    pixels wide (height scales proportionally) and labelled with the heading
    at capture time.

    Layout heuristic:
        cols = ceil(sqrt(n))
        rows = ceil(n / cols)
    e.g. 12 frames → 4×3, 15 frames → 4×4 (last row may be partial).
    """
    import io
    import math
    from PIL import Image as PILImage, ImageDraw, ImageFont

    if not jpeg_frames:
        raise ValueError("No frames to build mosaic from.")

    # Decode & resize all frames
    cells = []
    for raw in jpeg_frames:
        img = PILImage.open(io.BytesIO(raw))
        ratio = cell_width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((cell_width, new_h), PILImage.LANCZOS)
        cells.append(img)

    cell_h = cells[0].height  # assume all frames are the same size
    n = len(cells)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    label_h = 24  # pixels reserved for the heading label
    mosaic_w = cols * cell_width
    mosaic_h = rows * (cell_h + label_h)

    mosaic = PILImage.new("RGB", (mosaic_w, mosaic_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(mosaic)

    # Use default font (always available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, cell in enumerate(cells):
        col = idx % cols
        row = idx // cols
        x = col * cell_width
        y = row * (cell_h + label_h)

        # Paste frame
        mosaic.paste(cell, (x, y + label_h))

        # Draw label bar
        heading_str = f"{headings[idx]:.0f}°" if idx < len(headings) else f"#{idx+1}"
        label_text = f"[{idx+1}] {heading_str}"
        draw.rectangle([x, y, x + cell_width, y + label_h], fill=(50, 50, 50))
        draw.text((x + 6, y + 3), label_text, fill=(255, 255, 100), font=font)

    # Encode as JPEG
    buf = io.BytesIO()
    mosaic.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _build_heading_strip(
    jpeg_frames: list[bytes],
    headings: list[float],
    canvas_height: int = 480,
    hfov: float = _CAMERA_HFOV_DEG,
) -> bytes:
    """Build a panorama by assigning each output column to the nearest frame.

    **Nearest-center compositing** — for every output pixel column the
    *single* frame whose capture heading is closest is used.  Only a very
    narrow gradient (``SEAM_PX``) is applied at the boundary between two
    frames so there are no ghosting / double-exposure artefacts.

    When the total angular span exceeds 200° the panorama is split into
    **two rows** (top = first half, bottom = second half) so the VLM
    receives a more square image with larger visible objects.

    Heading tick marks and per-frame labels are drawn at the top of each
    row for spatial reference.

    Returns JPEG bytes.
    """
    import io
    import math
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFont

    if not jpeg_frames:
        raise ValueError("No frames to build heading strip from.")

    SEAM_PX = 8  # half-width of gradient transition at seams

    # --- Decode images ---
    images = []
    for raw in jpeg_frames:
        img = PILImage.open(io.BytesIO(raw))
        images.append(img)

    # Normalise headings to [0, 360) and sort left-to-right
    norm_headings = [(h % 360.0) for h in headings]
    order = sorted(range(len(norm_headings)), key=lambda i: norm_headings[i])
    sorted_hdg = [norm_headings[i] for i in order]
    sorted_imgs = [images[i] for i in order]

    n = len(sorted_imgs)
    span_start = sorted_hdg[0] - hfov / 2.0
    span_end = sorted_hdg[-1] + hfov / 2.0
    total_span = max(span_end - span_start, hfov)

    first_w, first_h = sorted_imgs[0].size
    px_per_deg = first_w / hfov
    row_w = int(total_span * px_per_deg)
    scale_y = canvas_height / first_h
    frame_w = int(first_w * scale_y)
    frame_h = canvas_height

    # --- Resize all frames once ---
    resized_arrs = []
    for img in sorted_imgs:
        resized_arrs.append(np.array(
            img.resize((frame_w, frame_h), PILImage.LANCZOS), dtype=np.uint8
        ))

    # --- Build single-row strip via nearest-center ---
    def _compose_strip(strip_w, strip_hdg, strip_arrs, strip_order, s_start):
        """Compose one horizontal strip.  Returns (uint8 array, draw-info)."""
        out = np.zeros((frame_h, strip_w, 3), dtype=np.uint8)
        m = len(strip_hdg)
        if m == 0:
            return out, []

        # For each output column, find nearest frame
        col_headings = np.linspace(s_start, s_start + strip_w / px_per_deg, strip_w)
        hdg_arr = np.array(strip_hdg)
        # |col_heading - frame_heading| → shape (strip_w, m)
        diffs = np.abs(col_headings[:, None] - hdg_arr[None, :])
        nearest = np.argmin(diffs, axis=1)  # index into strip arrays

        # Vector lookup: for each column, copy from the correct frame
        for fi in range(m):
            cols = np.where(nearest == fi)[0]
            if len(cols) == 0:
                continue
            centre_px = int((strip_hdg[fi] - s_start) * px_per_deg)
            for c in cols:
                src_x = int((c - centre_px) + frame_w // 2)
                src_x = max(0, min(frame_w - 1, src_x))
                out[:, c, :] = strip_arrs[fi][:, src_x, :]

        # Apply narrow seam blending at frame boundaries
        boundary_cols = []
        for c in range(1, strip_w):
            if nearest[c] != nearest[c - 1]:
                boundary_cols.append(c)

        for bc in boundary_cols:
            fi_left = nearest[max(0, bc - 1)]
            fi_right = nearest[min(strip_w - 1, bc)]
            blend_start = max(0, bc - SEAM_PX)
            blend_end = min(strip_w, bc + SEAM_PX)
            for bx in range(blend_start, blend_end):
                alpha = (bx - blend_start) / max(1, blend_end - blend_start - 1)
                # Get pixels from both frames
                cx_l = int((bx - int((strip_hdg[fi_left] - s_start) * px_per_deg)) + frame_w // 2)
                cx_r = int((bx - int((strip_hdg[fi_right] - s_start) * px_per_deg)) + frame_w // 2)
                cx_l = max(0, min(frame_w - 1, cx_l))
                cx_r = max(0, min(frame_w - 1, cx_r))
                pix_l = strip_arrs[fi_left][:, cx_l, :].astype(np.float32)
                pix_r = strip_arrs[fi_right][:, cx_r, :].astype(np.float32)
                out[:, bx, :] = np.clip(pix_l * (1 - alpha) + pix_r * alpha, 0, 255).astype(np.uint8)

        return out, strip_order

    # --- Decide layout: 1 row or 2 rows ---
    use_two_rows = total_span > 200.0 and n >= 6
    label_h = 30

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    if use_two_rows:
        mid = n // 2
        # Row 1: first half of frames
        r1_hdg = sorted_hdg[:mid]
        r1_arrs = resized_arrs[:mid]
        r1_order = order[:mid]
        r1_start = r1_hdg[0] - hfov / 2.0
        r1_end = r1_hdg[-1] + hfov / 2.0
        r1_w = int((r1_end - r1_start) * px_per_deg)

        # Row 2: second half of frames
        r2_hdg = sorted_hdg[mid:]
        r2_arrs = resized_arrs[mid:]
        r2_order = order[mid:]
        r2_start = r2_hdg[0] - hfov / 2.0
        r2_end = r2_hdg[-1] + hfov / 2.0
        r2_w = int((r2_end - r2_start) * px_per_deg)

        canvas_w = max(r1_w, r2_w)
        canvas_h = 2 * (frame_h + label_h)
        canvas = PILImage.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        # Compose row 1
        strip1, _ = _compose_strip(r1_w, r1_hdg, r1_arrs, r1_order, r1_start)
        canvas.paste(PILImage.fromarray(strip1), (0, label_h))
        _draw_heading_labels(draw, r1_hdg, r1_order, r1_start, r1_end,
                             px_per_deg, label_h, 0, canvas_w, font, font_sm)

        # Compose row 2
        y2 = frame_h + label_h
        strip2, _ = _compose_strip(r2_w, r2_hdg, r2_arrs, r2_order, r2_start)
        canvas.paste(PILImage.fromarray(strip2), (0, y2 + label_h))
        _draw_heading_labels(draw, r2_hdg, r2_order, r2_start, r2_end,
                             px_per_deg, label_h, y2, canvas_w, font, font_sm)
    else:
        # Single row
        canvas_w = row_w
        canvas_h = frame_h + label_h
        canvas = PILImage.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        strip, _ = _compose_strip(row_w, sorted_hdg, resized_arrs, order, span_start)
        canvas.paste(PILImage.fromarray(strip), (0, label_h))
        _draw_heading_labels(draw, sorted_hdg, order, span_start, span_end,
                             px_per_deg, label_h, 0, canvas_w, font, font_sm)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _draw_heading_labels(draw, sorted_hdg, order, span_start, span_end,
                         px_per_deg, label_h, y_offset, canvas_w, font, font_sm):
    """Draw heading ticks and frame markers for one panorama row."""
    import math

    # Background bar
    draw.rectangle([(0, y_offset), (canvas_w, y_offset + label_h)],
                   fill=(20, 20, 20))

    # Degree ticks every 30°
    tick_interval = 30.0
    tick = math.ceil(span_start / tick_interval) * tick_interval
    while tick <= span_end:
        tx = int((tick - span_start) * px_per_deg)
        if 0 <= tx < canvas_w:
            draw.line([(tx, y_offset + label_h - 8), (tx, y_offset + label_h)],
                      fill=(200, 200, 200), width=2)
            draw.text((tx + 3, y_offset + 2), f"{tick % 360:.0f}°",
                      fill=(255, 255, 100), font=font)
        tick += tick_interval

    # Frame markers
    for idx, hdg in enumerate(sorted_hdg):
        tx = int((hdg - span_start) * px_per_deg)
        if 0 <= tx < canvas_w:
            draw.line([(tx, y_offset + label_h - 12), (tx, y_offset + label_h)],
                      fill=(100, 255, 100), width=2)
            draw.text((tx + 3, y_offset + 14), f"[{order[idx]+1}]",
                      fill=(100, 255, 100), font=font_sm)


def _stitch_panorama(
    jpeg_frames: list[bytes],
    headings: list[float],
) -> tuple[bytes, str]:
    """Build a 360° panoramic image from ordered camera frames.

    Strategy (hybrid):
      1. **OpenCV Stitcher** — feature-based homography + multi-band blend.
         Produces the best result when sufficient texture is present.
      2. **Heading-based strip** — cylindrical projection using odometry.
         Always works; no feature matching required.
      3. **Grid mosaic** — labeled thumbnail grid (last resort).

    Returns:
        ``(jpeg_bytes, method)`` where *method* is one of
        ``'opencv_stitcher'``, ``'heading_strip'``, or ``'grid_mosaic'``.
    """
    import io
    import numpy as np

    if not jpeg_frames:
        raise ValueError("No frames to stitch.")

    # --- Attempt 1: OpenCV Stitcher ---
    try:
        import cv2

        cv_images = []
        for raw in jpeg_frames:
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                cv_images.append(img)

        if len(cv_images) >= 2:
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            status, pano = stitcher.stitch(cv_images)
            if status == cv2.Stitcher_OK and pano is not None:
                # Encode result as JPEG
                _, buf = cv2.imencode('.jpg', pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
                _stderr(f"OpenCV stitcher succeeded: {pano.shape[1]}×{pano.shape[0]}")
                return buf.tobytes(), 'opencv_stitcher'
            else:
                _stderr(f"OpenCV stitcher returned status {status} — falling back")
    except ImportError:
        _stderr("OpenCV not available — skipping feature-based stitcher")
    except Exception as exc:
        _stderr(f"OpenCV stitcher failed: {exc} — falling back")

    # --- Attempt 2: Heading-based cylindrical strip ---
    try:
        result = _build_heading_strip(jpeg_frames, headings)
        _stderr("Heading-based strip succeeded")
        return result, 'heading_strip'
    except Exception as exc:
        _stderr(f"Heading strip failed: {exc} — falling back to grid")

    # --- Attempt 3: Grid mosaic (always works) ---
    result = _build_grid_mosaic(jpeg_frames, headings)
    return result, 'grid_mosaic'


# =========================================================================
# ROTATE AND SCAN TOOL
# =========================================================================

@mcp.tool(
    title="Rotate and Scan (panoramic)",
    description=(
        "Perform a full 360° rotation and return a SINGLE stitched panoramic "
        "image of the entire surroundings.  The panorama is built by capturing "
        "frames at regular intervals and stitching them into one continuous "
        "wide image — NOT a grid of thumbnails.\n\n"
        "This is the **PRIMARY tool for search tasks**: one call gives the "
        "VLM a complete 360° view of the environment in a single image so "
        "nothing is missed.  Use it as Step 1 whenever you need to find or "
        "survey something.\n\n"
        "Stitching pipeline: OpenCV feature-based stitcher → heading-based "
        "cylindrical strip (fallback) → grid mosaic (last resort).\n\n"
        "Defaults: 360° sweep in 24° steps (= 15 frames with ~38° overlap "
        "given the 62° camera FOV). Negative total_angle sweeps clockwise.\n\n"
        "Returns: a text summary + ONE panoramic JPEG."
    ),
)
def rotate_and_scan(
    total_angle: float = 360.0,
    step_angle: float = 24.0,
    settle_time: float = 0.3,
    speed: float = 0.3,
):
    """Rotate and capture camera frames at each step, returning a panorama.

    Args:
        total_angle: Total rotation in degrees (positive=CCW, negative=CW).
                     Default 360° for a full sweep.
        step_angle:  Degrees per step.  Default 24° (15 steps for 360°,
                     each with ~38° overlap given the 62° camera FOV).
        settle_time: Seconds to wait after each step for the robot to
                     decelerate and the camera to publish a stable frame.
                     Default 0.3 s.
        speed:       Angular speed in rad/s.  Default 0.3 rad/s (slow for
                     accurate captures).
    """
    import time as _time
    import math as _math

    bridge = _get_bridge()

    # Positive total_angle → CCW (positive turn angles)
    # Negative total_angle → CW  (negative turn angles)
    sign = 1.0 if total_angle >= 0 else -1.0
    total_abs = abs(total_angle)
    step_abs = abs(step_angle)
    max_steps = max(1, int(round(total_abs / step_abs)))

    raw_frames: list[bytes] = []   # raw JPEG bytes for mosaic builder
    headings: list[float] = []     # yaw in degrees at each capture

    # Track total sweep via odometry so per-step overshoot is compensated
    odom0 = bridge.get_odom()
    sweep_start_yaw = _math.radians(odom0["orientation_yaw_deg"]) if odom0 else 0.0
    total_target_rad = _math.radians(total_abs)
    prev_yaw = sweep_start_yaw
    sweep_accumulated_rad = 0.0  # absolute value of total rotation so far

    direction_label = 'CCW' if sign > 0 else 'CW'
    _stderr(f"rotate_and_scan: up to {max_steps} steps × {step_abs}° {direction_label}")

    for i in range(max_steps):
        # Calculate how much rotation remains for the total sweep
        remaining_deg = total_abs - _math.degrees(sweep_accumulated_rad)
        if remaining_deg < 2.0:
            _stderr(f"  step {i+1}: sweep complete ({_math.degrees(sweep_accumulated_rad):.1f}° done)")
            break

        # Use the smaller of step_angle and remaining angle
        this_step_deg = min(step_abs, remaining_deg)

        # Turn one step (sign controls direction: positive=CCW, negative=CW)
        _stderr(f"  step {i+1}/{max_steps}: turning {sign * this_step_deg:.1f}° ...")
        result = bridge.turn(angle_deg=sign * this_step_deg, angular_speed=speed)
        _stderr(f"  step {i+1}/{max_steps}: turn result: {result}")

        # Use the actual angle from turn() result for sweep tracking
        # (more reliable than re-reading odometry separately)
        if result and result.get("success"):
            actual_deg = abs(result.get("angle_actual_deg", 0))
            sweep_accumulated_rad += _math.radians(actual_deg)
        elif result:
            _stderr(f"  step {i+1}/{max_steps}: turn FAILED: {result.get('message', '?')}")
            # Still update from odometry in case partial rotation happened
            odom_now = bridge.get_odom()
            if odom_now:
                cur_yaw = _math.radians(odom_now["orientation_yaw_deg"])
                delta = cur_yaw - prev_yaw
                while delta > _math.pi:
                    delta -= 2 * _math.pi
                while delta < -_math.pi:
                    delta += 2 * _math.pi
                sweep_accumulated_rad += abs(delta)
                prev_yaw = cur_yaw

        # Update prev_yaw for next iteration
        odom_now = bridge.get_odom()
        if odom_now:
            prev_yaw = _math.radians(odom_now["orientation_yaw_deg"])

        # Extra settle time for stable camera frame
        if settle_time > 0:
            _time.sleep(settle_time)

        # Wait for a genuinely fresh frame
        image_bytes, timestamp = bridge.wait_for_fresh_frame(timeout=1.5)

        if image_bytes is not None:
            raw_frames.append(image_bytes)
        else:
            _stderr(f"  step {i+1}/{max_steps}: no frame captured")

        # Record heading for the summary
        if odom_now:
            headings.append(round(odom_now["orientation_yaw_deg"], 1))
        else:
            headings.append(round(i * step_abs * sign, 1))

        _stderr(f"  step {i+1}/{max_steps}: heading={headings[-1]:.1f}° "
                f"swept={_math.degrees(sweep_accumulated_rad):.1f}° "
                f"frame={'OK' if image_bytes else 'MISS'}")

    if not raw_frames:
        raise ValueError(
            "rotate_and_scan captured 0 frames. "
            "Is the camera node running?"
        )

    # Build the stitched panorama (hybrid: OpenCV → heading strip → grid)
    pano_bytes, stitch_method = _stitch_panorama(raw_frames, headings)
    pano_image = Image(data=pano_bytes, format="jpeg")

    # Store panorama on bridge so the web viewer can display it
    bridge.set_mosaic(pano_bytes)

    swept_deg = _math.degrees(sweep_accumulated_rad)
    method_desc = {
        'opencv_stitcher': 'seamless feature-based panorama',
        'heading_strip': 'heading-based cylindrical panorama',
        'grid_mosaic': 'labeled grid mosaic (stitching unavailable)',
    }.get(stitch_method, stitch_method)

    summary = (
        f"Captured {len(raw_frames)} frames over a {swept_deg:.0f}° sweep "
        f"({step_abs:.0f}° steps, {direction_label}).\n"
        f"Stitch method: {method_desc}.\n"
        f"Headings (deg): {headings}\n\n"
        "The attached panoramic image shows a CONTINUOUS 360° view of the "
        "environment.  Scan the ENTIRE image from left to right — objects "
        "appear at their true heading positions.  If the target is visible, "
        "report its approximate heading (degrees) based on the tick marks "
        "at the top of the image and describe what you see."
    )

    return [summary, pano_image]


# =========================================================================
# HELPERS
# =========================================================================

def _json_default(obj):
    """Handle non-serialisable values (inf, nan) in JSON output."""
    import math
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
    return str(obj)


def _range_stats(ranges, start_idx, total, angle_start_deg, angle_end_deg):
    """Compute min/max/avg for a lidar angular slice (degrees, 0 = front)."""
    import math
    valid = []
    for i, r in enumerate(ranges):
        # Angle of this ray in degrees (0 = front, CCW positive)
        angle = (i / total) * 360.0
        # Normalise to [-180, 180] for easier quadrant math is not needed
        # since our quadrants are defined in [0, 360)
        in_range = False
        if angle_start_deg < 0:
            shifted_start = angle_start_deg % 360
            in_range = angle >= shifted_start or angle < angle_end_deg
        elif angle_end_deg > 360:
            in_range = angle >= angle_start_deg or angle < (angle_end_deg % 360)
        else:
            in_range = angle_start_deg <= angle < angle_end_deg

        if in_range and not math.isinf(r) and r > 0:
            valid.append(r)

    if not valid:
        return {"min_m": None, "max_m": None, "avg_m": None, "points": 0}

    return {
        "min_m": round(min(valid), 3),
        "max_m": round(max(valid), 3),
        "avg_m": round(sum(valid) / len(valid), 3),
        "points": len(valid),
    }


# =========================================================================
# SERVER ENTRY POINT
# =========================================================================

def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/sse",
    options: dict = {},
) -> None:
    """Start the TurtleBot3 MCP server.

    Options (from runner config YAML):
        ros_domain_id:  ROS_DOMAIN_ID to set before rclpy.init()
        camera_topic:   Camera compressed image topic (default: /camera/image/compressed)
        lidar_topic:    LiDAR scan topic (default: /scan)
        odom_topic:     Odometry topic (default: /odom)
        cmd_vel_topic:  Velocity command topic (default: /cmd_vel)
        camera_viewer_port:  Port for the MJPEG camera viewer (default: 18280, 0 to disable)
        camera_viewer_autostart:  Start viewer automatically (default: true)
    """
    global _bridge

    verbose = "verbose" in options
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Set ROS_DOMAIN_ID *before* importing rclpy / creating the bridge
    ros_domain_id = options.get("ros_domain_id")
    if ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)

    _stderr(f"PID={os.getpid()}  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(not set)')}")
    _stderr(f"options={options}")

    bridge_kwargs = {}
    for key in ("camera_topic", "lidar_topic", "odom_topic", "cmd_vel_topic"):
        if key in options:
            bridge_kwargs[key] = options[key]
    if ros_domain_id is not None:
        bridge_kwargs["ros_domain_id"] = ros_domain_id

    _stderr(f"bridge_kwargs={bridge_kwargs}")

    from .ros_bridge import TurtleBot3Bridge
    _bridge = TurtleBot3Bridge.get_instance(**bridge_kwargs)

    _stderr(f"Bridge initialized — camera={_bridge._camera_topic}  lidar={_bridge._lidar_topic}  odom={_bridge._odom_topic}")

    # Auto-start camera viewer if configured
    viewer_port = options.get("camera_viewer_port", 18280)
    autostart = options.get("camera_viewer_autostart", True)
    if autostart and viewer_port:
        try:
            url = _bridge.start_camera_viewer(port=int(viewer_port), open_browser=True)
            _stderr(f"Camera viewer auto-started at {url}")
        except Exception as e:
            _stderr(f"Camera viewer auto-start failed: {e}")

    logger.info(f"Starting TurtleBot3 MCP Server at {host}:{port}{path}")
    logger.info(
        "10 Tools: get_camera_image, get_lidar_scan, get_odometry, check_path_clear, "
        "move_forward, turn, navigate_safely, stop, open_camera_viewer, diagnose_ros"
    )

    if not verbose:
        import uvicorn.config
        uvicorn.config.LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    mcp.run(
        transport=transport,
        host=host,
        port=port,
        path=path,
        uvicorn_config={"access_log": False, "log_level": "warning"}
        if not verbose
        else {},
    )


if __name__ == "__main__":
    run()

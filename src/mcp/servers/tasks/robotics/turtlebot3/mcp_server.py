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
import threading
import time

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

_front_check_guard_lock = threading.Lock()
_front_check_guard = {
    "checks_since_motion": 0,
    "last_result": None,
    "last_threshold": None,
    "last_timestamp": 0.0,
}

FRONT_CHECK_MAX_WITHOUT_MOTION = 2
FRONT_CHECK_CACHE_TTL_S = 2.0


def _reset_front_check_guard() -> None:
    with _front_check_guard_lock:
        _front_check_guard["checks_since_motion"] = 0
        _front_check_guard["last_result"] = None
        _front_check_guard["last_threshold"] = None
        _front_check_guard["last_timestamp"] = 0.0


def _mark_motion_if_success(result: dict) -> None:
    """Reset front-check throttling after successful movement/rotation."""
    if not isinstance(result, dict):
        return

    status = str(result.get("status", "")).lower()
    if status in ("ok", "success", "completed", "stopped"):
        _reset_front_check_guard()


def _should_reuse_front_check(threshold_m: float) -> dict | None:
    """Return cached front check result when redundant checks should be throttled."""
    now = time.time()
    with _front_check_guard_lock:
        checks_since_motion = _front_check_guard["checks_since_motion"]
        last_result = _front_check_guard["last_result"]
        last_threshold = _front_check_guard["last_threshold"]
        last_timestamp = _front_check_guard["last_timestamp"]

    if checks_since_motion < FRONT_CHECK_MAX_WITHOUT_MOTION:
        return None
    if last_result is None or last_threshold is None:
        return None
    if abs(float(last_threshold) - float(threshold_m)) > 1e-6:
        return None
    if (now - float(last_timestamp)) > FRONT_CHECK_CACHE_TTL_S:
        return None

    reused = dict(last_result)
    reused["cached"] = True
    reused["suppressed_redundant_check"] = True
    reused["message"] = (
        "Reused recent front-clearance result to avoid redundant checks. "
        "Move or turn before additional front checks."
    )
    return reused


def _record_front_check_result(threshold_m: float, result: dict) -> None:
    with _front_check_guard_lock:
        _front_check_guard["checks_since_motion"] += 1
        _front_check_guard["last_result"] = result
        _front_check_guard["last_threshold"] = float(threshold_m)
        _front_check_guard["last_timestamp"] = time.time()


def _get_bridge():
    """Return the TurtleBot3Bridge singleton (must be initialised via run())."""
    global _bridge
    if _bridge is None:
        raise RuntimeError(
            "TurtleBot3Bridge not initialised. Start this server via run()."
        )
    return _bridge


def _decision_context(bridge, tool_name: str, args: dict) -> str:
    """Build a short human-readable rationale for why a tool was chosen."""
    try:
        odom = bridge.get_odom() or {}
        heading = odom.get("orientation_yaw_deg")
        lin_vel = (odom.get("linear_velocity") or {}).get("x")
        front_dist = bridge.check_obstacle(direction="front")

        if tool_name in ("move_forward", "navigate_safely"):
            if front_dist is None:
                return "forward motion requested; lidar front distance unavailable"
            if front_dist < 0.3:
                return f"path appears tight (front≈{front_dist:.2f}m), so cautious forward step"
            return f"front path appears clear (front≈{front_dist:.2f}m), advancing toward objective"

        if tool_name in ("check_path_clear", "get_lidar_scan"):
            if front_dist is None:
                return "need environment clearance estimate before movement"
            return f"safety/route check before action (front≈{front_dist:.2f}m)"

        if tool_name in ("turn", "turn_to_heading"):
            if heading is None:
                return "heading adjustment requested by navigation plan"
            return f"heading alignment for next step (current heading≈{heading:.1f}°)"

        if tool_name in ("scan_step", "rotate_and_scan", "look_for_target", "get_camera_image"):
            if lin_vel is not None and abs(lin_vel) > 0.02:
                return "visual update while robot is in motion"
            return "visual search/confirmation needed for target or scene understanding"

        if tool_name == "stop":
            return "stop requested for safety or completion condition"

        if tool_name == "get_odometry":
            return "pose/heading verification after movement decision"

        return "selected by current task plan"
    except Exception:
        return "selected by current task plan"


def _argument_context(bridge, tool_name: str, args: dict) -> dict:
    """Build per-argument rationale for operator-facing trace logs."""
    reasons = {}
    if not args:
        return reasons

    try:
        front_dist = bridge.check_obstacle(direction="front")
    except Exception:
        front_dist = None

    if tool_name in ("move_forward", "navigate_safely"):
        if "distance" in args:
            dist = args.get("distance")
            if front_dist is not None:
                if front_dist < 0.3:
                    reasons["distance"] = f"short step due to nearby obstacle (front≈{front_dist:.2f}m)"
                elif front_dist < 1.0:
                    reasons["distance"] = f"moderate step for partial clearance (front≈{front_dist:.2f}m)"
                else:
                    reasons["distance"] = f"longer stride is safe (front≈{front_dist:.2f}m)"
            else:
                reasons["distance"] = f"default/planned motion step ({dist})"
        if "speed" in args:
            reasons["speed"] = "chosen to balance stability and safety for indoor navigation"
        if "obstacle_threshold_m" in args:
            reasons["obstacle_threshold_m"] = "safety margin for triggering obstacle avoidance"

    elif tool_name == "check_path_clear":
        if "direction" in args:
            reasons["direction"] = "requested movement/planning direction to validate"
        if "threshold_m" in args:
            reasons["threshold_m"] = "minimum acceptable clearance before movement"

    elif tool_name in ("turn", "turn_to_heading"):
        if "angle" in args:
            reasons["angle"] = "relative heading correction needed for next navigation step"
        if "target_heading_deg" in args:
            reasons["target_heading_deg"] = "absolute heading from scan-based target/location estimate"
        if "speed" in args:
            reasons["speed"] = "moderate angular speed to reduce overshoot"

    elif tool_name in ("scan_step", "rotate_and_scan", "look_for_target"):
        if "step_angle" in args:
            reasons["step_angle"] = "scan resolution tradeoff: smaller is more precise, larger is faster"
        if "total_angle" in args:
            reasons["total_angle"] = "scan coverage requested by current search objective"
        if "sweep_width_deg" in args:
            reasons["sweep_width_deg"] = "focused search arc around likely target heading"

    elif tool_name == "open_camera_viewer" and "port" in args:
        reasons["port"] = "viewer server port for operator monitoring"

    return reasons


def _trace_tool_call(tool_name: str, reason: str, args: dict | None = None) -> None:
    try:
        bridge = _get_bridge()
        payload_args = args or {}
        why = _decision_context(bridge, tool_name=tool_name, args=payload_args)
        arg_reasons = _argument_context(bridge, tool_name=tool_name, args=payload_args)
        bridge.record_tool_call(
            tool_name=tool_name,
            reason=f"{reason} Why: {why}",
            args=payload_args,
            arg_reasons=arg_reasons,
        )
    except Exception:
        pass


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
    _trace_tool_call("get_camera_image", "Capture latest scene for perception/update.")
    bridge = _get_bridge()
    # wait_for_fresh_frame blocks until a frame newer than the current
    # cached one arrives, avoiding stale / motion-blurred images.
    image_bytes, timestamp = bridge.wait_for_fresh_frame(
        timeout=1.0,
        min_new_frames=1,
        poll_interval=0.03,
    )

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
    _trace_tool_call("get_lidar_scan", "Read obstacle distances around the robot.", {"summarize": summarize})
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
    _trace_tool_call("get_odometry", "Read robot pose/velocity for localization feedback.")
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

    _trace_tool_call("move_forward", "Advance toward goal in controlled forward motion.", {
        "distance": distance,
        "speed": speed,
    })
    bridge = _get_bridge()
    result = bridge.move_forward(distance_m=distance, speed=speed)
    _mark_motion_if_success(result)
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

    _trace_tool_call("turn", "Adjust heading with relative rotation.", {
        "angle": angle,
        "speed": speed,
    })
    bridge = _get_bridge()
    result = bridge.turn(angle_deg=angle, angular_speed=speed)
    _mark_motion_if_success(result)
    return json.dumps(result, indent=2, default=_json_default)


@mcp.tool(
    title="Turn to Heading",
    description=(
        "Turn the TurtleBot3 to face an absolute heading in degrees. "
        "Unlike the 'turn' tool which takes a relative angle, this tool "
        "takes an absolute heading (e.g., 90° for East) and automatically "
        "computes the shortest rotation path, handling 0°/360° wrap-around. "
        "After the initial turn, it checks the residual error and performs "
        "up to 2 corrective micro-turns if needed (tolerance: 3°). "
        "Use this after rotate_and_scan to face the heading where a target "
        "was spotted."
    ),
)
def turn_to_heading(
    target_heading_deg: float,
) -> str:
    """Turn to face an absolute heading.

    Args:
        target_heading_deg: Desired heading in degrees (e.g., 0=North, 90=East,
            180=South, 270=West). Uses the same coordinate frame as odometry
            and rotate_and_scan headings.
    """
    _trace_tool_call("turn_to_heading", "Face target heading from scan/navigation plan.", {
        "target_heading_deg": target_heading_deg,
    })
    bridge = _get_bridge()
    result = bridge.turn_to_heading(target_deg=target_heading_deg)
    _mark_motion_if_success(result)
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
    _trace_tool_call("stop", "Immediate halt for safety or task completion.")
    bridge = _get_bridge()
    bridge.stop()
    _reset_front_check_guard()
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

    _trace_tool_call("check_path_clear", "Verify clearance before motion step.", {
        "direction": direction,
        "threshold_m": threshold_m,
    })

    if direction == "front":
        reused = _should_reuse_front_check(threshold_m=threshold_m)
        if reused is not None:
            return json.dumps(reused, indent=2, default=_json_default)

    bridge = _get_bridge()
    result = bridge.check_path_clear(direction=direction, threshold_m=threshold_m)

    if direction == "front" and isinstance(result, dict):
        _record_front_check_result(threshold_m=threshold_m, result=result)

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

    _trace_tool_call("navigate_safely", "Move with automatic obstacle avoidance.", {
        "distance": distance,
        "speed": speed,
        "obstacle_threshold_m": obstacle_threshold_m,
    })
    bridge = _get_bridge()
    result = bridge.navigate_safely(
        distance_m=distance,
        speed=speed,
        obstacle_threshold_m=obstacle_threshold_m,
    )
    _mark_motion_if_success(result)
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
    _trace_tool_call("open_camera_viewer", "Open operator visual monitor for robot view.", {"port": port})
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
    _trace_tool_call("diagnose_ros", "Collect ROS connectivity/health diagnostics.")
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


def _heading_to_cardinal(deg: float) -> str:
    """Convert heading degrees to cardinal/intercardinal label."""
    deg = deg % 360.0
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((deg + 22.5) / 45.0) % 8
    return dirs[idx]


def _build_contact_sheet(
    jpeg_frames: list[bytes],
    headings: list[float],
    cell_width: int = 540,
    padding: int = 6,
) -> bytes:
    """Build an annotated contact-sheet grid from captured frames.

    Each frame is displayed as an individual sharp thumbnail sorted by
    heading (left-to-right, top-to-bottom) with a prominent label showing
    the frame number, heading angle, and cardinal direction.

    This avoids all stitching artefacts (seams, ghosting, exposure
    mismatch) and gives the VLM individually crisp frames it can reason
    about ("frame [5] at 90° E shows a chair").

    Grid layout targets a roughly 16:9 aspect ratio.

    Returns JPEG bytes.
    """
    import io
    import math
    from PIL import Image as PILImage, ImageDraw, ImageFont

    if not jpeg_frames:
        raise ValueError("No frames to build contact sheet from.")

    n = len(jpeg_frames)

    # --- Sort frames by heading ---
    norm_hdg = [(h % 360.0) for h in headings]
    order = sorted(range(n), key=lambda i: norm_hdg[i])
    sorted_jpegs = [jpeg_frames[i] for i in order]
    sorted_hdg = [norm_hdg[i] for i in order]

    # --- Decode & resize thumbnails ---
    cells = []
    for raw in sorted_jpegs:
        img = PILImage.open(io.BytesIO(raw))
        ratio = cell_width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((cell_width, new_h), PILImage.LANCZOS)
        cells.append(img)

    cell_h = cells[0].height

    # --- Pick grid dimensions targeting ~16:9 ---
    # Try different column counts and pick the one closest to 16:9
    best_cols = max(1, math.ceil(math.sqrt(n)))
    best_ratio_err = float('inf')
    for c in range(max(1, n // 6), min(n + 1, n // 1 + 1)):
        r = math.ceil(n / c)
        w = c * (cell_width + padding) + padding
        h = r * (cell_h + 30 + padding) + padding  # 30 for label
        ratio = w / max(h, 1)
        err = abs(ratio - 16.0 / 9.0)
        if err < best_ratio_err:
            best_ratio_err = err
            best_cols = c

    cols = best_cols
    rows = math.ceil(n / cols)

    label_h = 30
    total_cell_h = cell_h + label_h
    sheet_w = cols * (cell_width + padding) + padding
    sheet_h = rows * (total_cell_h + padding) + padding

    sheet = PILImage.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for idx in range(n):
        col = idx % cols
        row = idx // cols
        x = padding + col * (cell_width + padding)
        y = padding + row * (total_cell_h + padding)

        # Label bar background
        draw.rectangle(
            [x, y, x + cell_width, y + label_h], fill=(50, 50, 60))

        # Frame number + heading + cardinal
        hdg = sorted_hdg[idx]
        orig_idx = order[idx] + 1  # 1-based capture order
        cardinal = _heading_to_cardinal(hdg)
        label = f"[{orig_idx}]  {hdg:.0f}°  {cardinal}"
        draw.text((x + 8, y + 5), label,
                  fill=(255, 255, 100), font=font)

        # Paste thumbnail
        sheet.paste(cells[idx], (x, y + label_h))

        # Thin border around cell
        draw.rectangle(
            [x - 1, y - 1, x + cell_width + 1, y + total_cell_h + 1],
            outline=(80, 80, 80), width=1)

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _stitch_panorama(
    jpeg_frames: list[bytes],
    headings: list[float],
) -> tuple[bytes, str]:
    """Build a 360° survey image from ordered camera frames.

    Strategy:
      1. **OpenCV Stitcher** — feature-based homography + multi-band blend.
         Produces a seamless panorama when sufficient texture is present.
      2. **Contact sheet** — annotated grid of individually sharp frames
         sorted by heading.  Always works, no artefacts, and the VLM can
         reason about each frame independently.

    Returns:
        ``(jpeg_bytes, method)`` where *method* is one of
        ``'opencv_stitcher'`` or ``'contact_sheet'``.
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
                _, buf = cv2.imencode('.jpg', pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
                _stderr(f"OpenCV stitcher succeeded: {pano.shape[1]}×{pano.shape[0]}")
                return buf.tobytes(), 'opencv_stitcher'
            else:
                _stderr(f"OpenCV stitcher returned status {status} — falling back")
    except ImportError:
        _stderr("OpenCV not available — skipping feature-based stitcher")
    except Exception as exc:
        _stderr(f"OpenCV stitcher failed: {exc} — falling back")

    # --- Attempt 2: Contact sheet (always works) ---
    result = _build_contact_sheet(jpeg_frames, headings)
    _stderr("Contact sheet built successfully")
    return result, 'contact_sheet'


# =========================================================================
# ROTATE AND SCAN TOOL
# =========================================================================

@mcp.tool(
    title="Scan Step (progressive search)",
    description=(
        "Turn a small angle and return ONE fresh camera frame plus heading. "
        "Use this for progressive search loops where you must stop as soon "
        "as the target becomes visible. This tool is low-latency and intended "
        "for visibility-gated scan/confirm behavior."
    ),
)
def scan_step(
    step_angle: float = 12.0,
    settle_time: float = 0.1,
    speed: float = 0.35,
    frame_timeout: float = 0.8,
    stable_frames: int = 1,
):
    """Rotate a small step, then capture a fresh frame and heading.

    Args:
        step_angle: Relative turn in degrees (positive=CCW, negative=CW).
        settle_time: Optional extra wait after turn before frame capture.
        speed: Angular speed in rad/s.
        frame_timeout: Seconds to wait for a fresh frame.
        stable_frames: Number of distinct new frames to wait for (1 or 2).
    """
    import time as _time

    _trace_tool_call("scan_step", "Progressive search step with immediate visual check.", {
        "step_angle": step_angle,
        "settle_time": settle_time,
        "speed": speed,
    })
    bridge = _get_bridge()

    turn_result = bridge.turn(angle_deg=step_angle, angular_speed=speed)

    if settle_time > 0:
        _time.sleep(settle_time)

    image_bytes, timestamp = bridge.wait_for_fresh_frame(
        timeout=max(0.2, frame_timeout),
        min_new_frames=max(1, int(stable_frames)),
        poll_interval=0.03,
    )

    odom = bridge.get_odom()
    heading = round(odom["orientation_yaw_deg"], 1) if odom else None

    if image_bytes is None:
        payload = {
            "success": bool(turn_result and turn_result.get("success")),
            "heading_deg": heading,
            "turn": turn_result,
            "frame": "missing",
            "timestamp": timestamp,
        }
        return json.dumps(payload, default=_json_default)

    try:
        bridge.add_search_frame(image_bytes, heading if heading is not None else 0.0)
    except Exception:
        pass

    summary = {
        "success": bool(turn_result and turn_result.get("success")),
        "heading_deg": heading,
        "turn": turn_result,
        "frame": "ok",
        "timestamp": timestamp,
    }
    return [json.dumps(summary, default=_json_default), Image(data=image_bytes, format="jpeg")]

@mcp.tool(
    title="Rotate and Scan (360° survey)",
    description=(
        "Perform a full 360° rotation and return a SINGLE image surveying "
        "the entire surroundings.  Frames are captured at regular angular "
        "steps and assembled into an annotated contact-sheet grid sorted "
        "by heading — each frame is individually sharp and labelled with "
        "its heading and cardinal direction.\n\n"
        "This is the **PRIMARY tool for search tasks**: one call gives   "
        "a complete 360° view of the environment so nothing is missed. "
        "Use it as Step 1 whenever you need to find or survey something."
        "\n\n"
        "Pipeline: tries OpenCV feature-based panorama first; falls back "
        "to the contact-sheet grid (no stitching artefacts).\n\n"
        "Defaults: 360° sweep in 24° steps (= 15 frames with ~38° overlap "
        "given the 62° camera FOV). Negative total_angle sweeps clockwise."
        "\n\nReturns: a text summary + ONE survey image."
    ),
)
def rotate_and_scan(
    total_angle: float = 360.0,
    step_angle: float = 24.0,
    settle_time: float = 0.2,
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

    _trace_tool_call("rotate_and_scan", "Perform full-area visual survey for targets/obstacles.", {
        "total_angle": total_angle,
        "step_angle": step_angle,
    })
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
        image_bytes, timestamp = bridge.wait_for_fresh_frame(
            timeout=1.0,
            min_new_frames=1,
            poll_interval=0.03,
        )

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
        'contact_sheet': 'annotated contact-sheet grid',
    }.get(stitch_method, stitch_method)

    summary = (
        f"Captured {len(raw_frames)} frames over a {swept_deg:.0f}° sweep "
        f"({step_abs:.0f}° steps, {direction_label}).\n"
        f"Layout: {method_desc}.\n"
        f"Headings (deg): {headings}\n\n"
        "The attached image is a CONTACT SHEET — a grid of individually "
        "sharp frames sorted by heading (left-to-right, top-to-bottom).  "
        "Each frame is labelled with its capture-order number [N], heading "
        "in degrees, and cardinal direction (N/NE/E/SE/S/SW/W/NW).\n\n"
        "Examine EVERY frame carefully.  If the target is visible, report "
        "which frame number [N] it appears in, the heading, and describe "
        "what you see.  If not visible, say so."
    )

    return [summary, pano_image]


# =========================================================================
# TARGETED RE-ACQUISITION TOOL
# =========================================================================

@mcp.tool(
    title="Look for Target (targeted sweep)",
    description=(
        "Perform a targeted sweep around an expected heading to re-acquire "
        "a target that was spotted during rotate_and_scan.  The robot first "
        "turns so the sweep is CENTERED on the given heading, then captures "
        "full-resolution frames over a narrow arc (default 90°).\n\n"
        "Use this AFTER rotate_and_scan when you know approximately where "
        "the target is but need higher-resolution confirmation, or when "
        "get_camera_image didn't show the target after turning to the "
        "expected heading.\n\n"
        "Returns: a text summary + ONE contact-sheet image with "
        "full-resolution frames (no thumbnail downscaling)."
    ),
)
def look_for_target(
    heading_deg: float,
    sweep_width_deg: float = 90.0,
    step_angle: float = 15.0,
    settle_time: float = 0.3,
    speed: float = 0.3,
):
    """Targeted sweep around an expected heading with full-resolution frames.

    Args:
        heading_deg:    Center heading of the sweep in degrees (from
                        rotate_and_scan contact-sheet labels).
        sweep_width_deg: Total arc to sweep in degrees (default 90°).
                        The sweep spans heading ± sweep_width/2.
        step_angle:     Degrees per step (default 15°, finer than rotate_and_scan).
        settle_time:    Seconds to wait after each step (default 0.3).
        speed:          Angular speed in rad/s (default 0.3).
    """
    import time as _time
    import math as _math

    _trace_tool_call("look_for_target", "Re-acquire target in focused heading window.", {
        "heading_deg": heading_deg,
        "sweep_width_deg": sweep_width_deg,
    })
    bridge = _get_bridge()

    # --- Position at sweep start: heading - sweep_width/2 ---
    start_heading = heading_deg - sweep_width_deg / 2.0
    _stderr(f"look_for_target: centering on {heading_deg}°, "
            f"sweeping {sweep_width_deg}° ({start_heading:.0f}° → "
            f"{heading_deg + sweep_width_deg / 2.0:.0f}°)")

    pos_result = bridge.turn_to_heading(target_deg=start_heading)
    if not pos_result.get("success"):
        _stderr(f"look_for_target: failed to reach start heading: {pos_result}")
        # Continue anyway — partial sweep is better than nothing

    # --- Sweep (reuses the same logic as rotate_and_scan) ---
    sign = 1.0  # always sweep CCW (positive direction)
    total_abs = abs(sweep_width_deg)
    step_abs = abs(step_angle)
    max_steps = max(1, int(round(total_abs / step_abs)))

    raw_frames: list[bytes] = []
    headings: list[float] = []

    odom0 = bridge.get_odom()
    sweep_accumulated_rad = 0.0
    prev_yaw = _math.radians(odom0["orientation_yaw_deg"]) if odom0 else 0.0

    # Capture first frame at the start position
    image_bytes, _ = bridge.wait_for_fresh_frame(
        timeout=1.0,
        min_new_frames=1,
        poll_interval=0.03,
    )
    if image_bytes is not None:
        raw_frames.append(image_bytes)
        odom_now = bridge.get_odom()
        headings.append(round(odom_now["orientation_yaw_deg"], 1) if odom_now else start_heading)

    for i in range(max_steps):
        remaining_deg = total_abs - _math.degrees(sweep_accumulated_rad)
        if remaining_deg < 2.0:
            break

        this_step_deg = min(step_abs, remaining_deg)
        result = bridge.turn(angle_deg=sign * this_step_deg, angular_speed=speed)

        if result and result.get("success"):
            actual_deg = abs(result.get("angle_actual_deg", 0))
            sweep_accumulated_rad += _math.radians(actual_deg)
        elif result:
            odom_now = bridge.get_odom()
            if odom_now:
                cur_yaw = _math.radians(odom_now["orientation_yaw_deg"])
                delta = cur_yaw - prev_yaw
                while delta > _math.pi:
                    delta -= 2 * _math.pi
                while delta < -_math.pi:
                    delta += 2 * _math.pi
                sweep_accumulated_rad += abs(delta)

        odom_now = bridge.get_odom()
        if odom_now:
            prev_yaw = _math.radians(odom_now["orientation_yaw_deg"])

        if settle_time > 0:
            _time.sleep(settle_time)

        image_bytes, _ = bridge.wait_for_fresh_frame(
            timeout=1.0,
            min_new_frames=1,
            poll_interval=0.03,
        )
        if image_bytes is not None:
            raw_frames.append(image_bytes)
        else:
            _stderr(f"  look_for_target step {i+1}/{max_steps}: no frame captured")

        if odom_now:
            headings.append(round(odom_now["orientation_yaw_deg"], 1))
        else:
            headings.append(round(start_heading + i * step_abs, 1))

    if not raw_frames:
        raise ValueError(
            "look_for_target captured 0 frames. Is the camera node running?"
        )

    # Build contact sheet at FULL resolution (no downscaling)
    sheet_bytes = _build_contact_sheet(raw_frames, headings, cell_width=640)
    sheet_image = Image(data=sheet_bytes, format="jpeg")

    bridge.set_mosaic(sheet_bytes)

    swept_deg = _math.degrees(sweep_accumulated_rad)
    summary = (
        f"Targeted sweep: captured {len(raw_frames)} full-resolution frames "
        f"over {swept_deg:.0f}° centered on heading {heading_deg:.0f}°.\n"
        f"Headings (deg): {headings}\n\n"
        "This is a FULL-RESOLUTION contact sheet — each frame is at camera "
        "native resolution for maximum detail.  Examine every frame carefully "
        "for the target object.  Report which frame [N] and heading shows "
        "the target, or confirm it is not visible."
    )

    return [summary, sheet_image]


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
        "11 Tools: get_camera_image, get_lidar_scan, get_odometry, check_path_clear, "
        "move_forward, turn, navigate_safely, stop, open_camera_viewer, diagnose_ros, scan_step"
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

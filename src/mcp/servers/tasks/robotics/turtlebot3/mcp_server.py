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
# ROTATE AND SCAN TOOL
# =========================================================================

@mcp.tool(
    title="Rotate and Scan",
    description=(
        "Slowly rotate the robot while capturing a stable camera frame at "
        "every step. Returns ALL captured frames so you can examine them in "
        "one go. This is far more reliable than issuing individual turn + "
        "get_camera_image calls because it guarantees a settling delay and "
        "a fresh (non-blurred) frame at every heading.\n\n"
        "Use this for 'find object' / search tasks instead of manual loops.\n\n"
        "Defaults: 360° sweep in 30° steps (= 12 frames with ~32° overlap "
        "given the 62° camera FOV).  Negative total_angle sweeps clockwise."
    ),
)
def rotate_and_scan(
    total_angle: float = 360.0,
    step_angle: float = 30.0,
    settle_time: float = 0.3,
    speed: float = 0.3,
) -> list:
    """Rotate and capture camera frames at each step.

    Args:
        total_angle: Total rotation in degrees (positive=CCW, negative=CW).
                     Default 360° for a full sweep.
        step_angle:  Degrees per step.  Default 30° (12 steps for 360°).
        settle_time: Seconds to wait after each step for the robot to
                     decelerate and the camera to publish a stable frame.
                     Default 0.3 s.
        speed:       Angular speed in rad/s.  Default 0.3 rad/s.
    """
    import time as _time

    bridge = _get_bridge()

    direction = -1.0 if total_angle >= 0 else 1.0   # default sweep = CW
    total_abs = abs(total_angle)
    step_abs = abs(step_angle)
    steps = max(1, int(round(total_abs / step_abs)))

    frames = []  # list[Image]
    headings = []  # list[float] — yaw in degrees at each capture

    for i in range(steps):
        # Turn one step
        bridge.turn(angle_deg=direction * step_abs, angular_speed=speed)

        # Extra settle time (on top of the 0.3 s inside turn())
        if settle_time > 0:
            _time.sleep(settle_time)

        # Wait for a genuinely fresh frame
        image_bytes, timestamp = bridge.wait_for_fresh_frame(timeout=1.0)

        if image_bytes is not None:
            frames.append(Image(data=image_bytes, format="jpeg"))

        # Record heading for the summary
        odom = bridge.get_odom()
        if odom:
            headings.append(round(odom["orientation_yaw_deg"], 1))

    if not frames:
        raise ValueError(
            "rotate_and_scan captured 0 frames. "
            "Is the camera node running?"
        )

    # Return images directly — FastMCP serialises the list for the VLM.
    # Also prepend a text summary so the model knows which heading each
    # image corresponds to.
    summary = (
        f"Captured {len(frames)} frames over a {total_abs}° sweep "
        f"({step_abs}° steps, {'CCW' if direction == 1.0 else 'CW'}).\n"
        f"Headings (deg): {headings}\n"
        "Examine each image carefully for the target object."
    )
    # FastMCP allows returning mixed content; put text first, then images
    return [summary] + frames


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

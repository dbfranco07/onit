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
import os
import sys
import time
import threading
import logging
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

logger = logging.getLogger(__name__)


def _stderr(msg: str):
    """Print diagnostic to stderr so it isn't swallowed by multiprocessing."""
    print(f"[TurtleBot3Bridge] {msg}", file=sys.stderr, flush=True)

# TurtleBot3 Burger physical limits (ROBOTIS spec)
MAX_LINEAR_SPEED = 0.22   # m/s
MAX_ANGULAR_SPEED = 2.84  # rad/s

# Default timeout for motion commands
DEFAULT_MOTION_TIMEOUT = 30.0  # seconds

# Default control loop rate
CONTROL_HZ = 10  # Hz

# Default camera viewer port
DEFAULT_VIEWER_PORT = 18280

# -----------------------------------------------------------------------
# MJPEG Camera Viewer — streams live camera feed to the browser
# -----------------------------------------------------------------------

_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>TurtleBot3 Camera Viewer</title>
  <style>
    body {{
      margin: 0; background: #1e1e1e; color: #ccc;
      font-family: system-ui, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh;
    }}
    h1 {{ margin: 16px 0 8px; font-size: 1.3em; color: #eee; }}
    #status {{ font-size: 0.85em; margin-bottom: 8px; }}
    img {{
      max-width: 95vw; max-height: 80vh;
      border: 2px solid #444; border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>&#x1F916; TurtleBot3 Camera</h1>
  <div id="status">Waiting for frames&hellip;</div>
  <img id="cam" src="/stream" alt="Camera feed" />
  <script>
    const img = document.getElementById('cam');
    const st  = document.getElementById('status');
    let frames = 0;
    img.onload  = () => {{ frames++; st.textContent = 'Live — frame ' + frames; }};
    img.onerror = () => {{ st.textContent = 'Stream interrupted — retrying&hellip;'; setTimeout(() => {{ img.src = '/stream?' + Date.now(); }}, 1000); }};
  </script>
</body>
</html>
"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves an HTML page at / and MJPEG stream at /stream."""

    # Set by the factory function
    bridge_ref = None

    def log_message(self, format, *args):  # noqa: A002
        # Silence per-request logs
        pass

    def do_GET(self):
        if self.path == '/' or self.path.startswith('/index'):
            self._serve_html()
        elif self.path.startswith('/stream'):
            self._serve_mjpeg()
        elif self.path == '/snapshot':
            self._serve_snapshot()
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
                time.sleep(0.066)  # ~15 fps cap
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
                webbrowser.open(url)
            return url

        port = port or DEFAULT_VIEWER_PORT

        # Create a handler class bound to this bridge instance
        handler = type('BoundMJPEGHandler', (_MJPEGHandler,), {'bridge_ref': self})

        try:
            server = HTTPServer(('0.0.0.0', port), handler)
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
                webbrowser.open(url)
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
    # Public motion API
    # ------------------------------------------------------------------

    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        """Publish a Twist message on /cmd_vel."""
        twist = self._Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(twist)

    def stop(self):
        """Emergency stop — publish zero velocity."""
        self._publish_twist(0.0, 0.0)
        logger.info("STOP command sent")

    def _wait_for_odom(self, timeout=5.0):
        """Block until at least one odometry message has been received."""
        start = time.monotonic()
        while self.get_odom() is None:
            if time.monotonic() - start > timeout:
                return False
            time.sleep(0.05)
        return True

    def move_forward(self, distance_m=0.5, speed=0.2, timeout=None):
        """Move forward ``distance_m`` metres at ``speed`` m/s.

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

                self._publish_twist(linear_x=speed)
                time.sleep(rate_sleep)
        except Exception as e:
            self.stop()
            return {"success": False, "message": f"Error during move: {e}"}

    def turn(self, angle_deg=90.0, angular_speed=0.5, timeout=None):
        """Turn ``angle_deg`` degrees (positive = CCW, negative = CW).

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
                    return {
                        "success": True,
                        "angle_requested_deg": angle_deg,
                        "angle_actual_deg": round(math.degrees(accumulated), 2),
                        "start_yaw_deg": round(math.degrees(start_yaw), 2),
                        "end_yaw_deg": round(cur["orientation_yaw_deg"], 2),
                        "message": "Target angle reached",
                    }

                self._publish_twist(angular_z=direction * angular_speed)
                time.sleep(rate_sleep)
        except Exception as e:
            self.stop()
            return {"success": False, "message": f"Error during turn: {e}"}

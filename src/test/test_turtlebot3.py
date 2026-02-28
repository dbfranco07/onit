"""
Tests for TurtleBot3 MCP tools and ROS 2 bridge.

All ROS 2 dependencies (rclpy, sensor_msgs, nav_msgs, geometry_msgs) are
mocked so these tests can run without a ROS 2 installation.
"""

import base64
import json
import math
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Mock ROS 2 packages before importing any onit robotics code
# ---------------------------------------------------------------------------

# Create mock ROS message types
_mock_compressed_image = MagicMock()
_mock_laser_scan = MagicMock()
_mock_odometry = MagicMock()
_mock_twist = MagicMock()


def _create_mock_rclpy():
    """Build a complete mock rclpy module hierarchy."""
    rclpy = MagicMock()
    rclpy.ok.return_value = True

    # Mock Node
    mock_node = MagicMock()
    mock_node.create_subscription = MagicMock()
    mock_node.create_publisher = MagicMock(return_value=MagicMock())
    mock_node.destroy_node = MagicMock()
    rclpy.node.Node.return_value = mock_node

    # Mock QoS
    rclpy.qos.QoSProfile = MagicMock()
    rclpy.qos.ReliabilityPolicy.BEST_EFFORT = "best_effort"
    rclpy.qos.ReliabilityPolicy.RELIABLE = "reliable"
    rclpy.qos.HistoryPolicy.KEEP_LAST = "keep_last"
    rclpy.qos.DurabilityPolicy.VOLATILE = "volatile"

    # Spin should do nothing (we don't want blocking in tests)
    rclpy.spin = MagicMock()
    rclpy.init = MagicMock()

    # spin_once needs a small sleep to avoid busy-waiting in daemon thread
    def _mock_spin_once(node, timeout_sec=0.1):
        time.sleep(0.01)
    rclpy.spin_once = _mock_spin_once

    return rclpy, mock_node


_mock_rclpy, _mock_node = _create_mock_rclpy()

# Patch sys.modules before importing the bridge
sys.modules["rclpy"] = _mock_rclpy
sys.modules["rclpy.node"] = _mock_rclpy.node
sys.modules["rclpy.qos"] = _mock_rclpy.qos
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["nav_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()

# Make Twist() mock return an object with linear.x and angular.z
_mock_twist_instance = MagicMock()
_mock_twist_instance.linear.x = 0.0
_mock_twist_instance.angular.z = 0.0
sys.modules["geometry_msgs.msg"].Twist = MagicMock(return_value=_mock_twist_instance)

# Now safe to import
from src.mcp.servers.tasks.robotics.turtlebot3.ros_bridge import (
    TurtleBot3Bridge,
    _euler_from_quaternion,
    _normalize_angle,
    MAX_LINEAR_SPEED,
    MAX_ANGULAR_SPEED,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_bridge():
    """Reset singleton before each test."""
    TurtleBot3Bridge._instance = None
    yield
    # Properly shut down the bridge to stop the spin thread
    if TurtleBot3Bridge._instance is not None:
        try:
            TurtleBot3Bridge._instance._shutdown = True
            TurtleBot3Bridge._instance.stop_camera_viewer()
        except Exception:
            pass
    TurtleBot3Bridge._instance = None


@pytest.fixture
def bridge():
    """Create a TurtleBot3Bridge instance with mocked ROS 2."""
    # Patch warmup to be instant (mocked ROS never delivers real data)
    with patch.object(TurtleBot3Bridge, '_warmup', return_value=None):
        b = TurtleBot3Bridge.get_instance(
            camera_topic="/camera/image/compressed",
            lidar_topic="/scan",
            odom_topic="/odom",
            cmd_vel_topic="/cmd_vel",
            ros_domain_id=0,
        )
    return b


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

class TestEulerFromQuaternion:
    def test_identity(self):
        """Identity quaternion should give zero yaw."""
        _, _, yaw = _euler_from_quaternion(0, 0, 0, 1)
        assert abs(yaw) < 1e-6

    def test_90_deg_yaw(self):
        """90° yaw (CCW around z)."""
        # quaternion for 90° about z: (0, 0, sin(45°), cos(45°))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        _, _, yaw = _euler_from_quaternion(0, 0, s, c)
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_180_deg_yaw(self):
        """180° yaw."""
        _, _, yaw = _euler_from_quaternion(0, 0, 1, 0)
        assert abs(abs(yaw) - math.pi) < 1e-6


class TestNormalizeAngle:
    def test_in_range(self):
        assert abs(_normalize_angle(1.0) - 1.0) < 1e-9

    def test_wrap_positive(self):
        result = _normalize_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_wrap_negative(self):
        result = _normalize_angle(-3 * math.pi)
        assert -math.pi <= result <= math.pi


# ---------------------------------------------------------------------------
# Bridge singleton
# ---------------------------------------------------------------------------

class TestBridgeSingleton:
    def test_singleton(self, bridge):
        b2 = TurtleBot3Bridge.get_instance()
        assert bridge is b2

    def test_reset(self, bridge):
        TurtleBot3Bridge.reset_instance()
        assert TurtleBot3Bridge._instance is None


# ---------------------------------------------------------------------------
# Sensor getters (before any callbacks)
# ---------------------------------------------------------------------------

class TestSensorsNoData:
    def test_no_image(self, bridge):
        data, ts = bridge.get_image()
        assert data is None
        assert ts is None

    def test_no_lidar(self, bridge):
        assert bridge.get_lidar() is None

    def test_no_odom(self, bridge):
        assert bridge.get_odom() is None


# ---------------------------------------------------------------------------
# Sensor getters (with simulated callbacks)
# ---------------------------------------------------------------------------

class TestSensorsWithData:
    def _make_stamp(self, sec=1000, nanosec=0):
        stamp = MagicMock()
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def test_camera_callback(self, bridge):
        msg = MagicMock()
        msg.data = b"\xff\xd8\xff\xe0fake_jpeg_data"
        msg.header.stamp = self._make_stamp()

        bridge._camera_cb(msg)
        data, ts = bridge.get_image()
        assert data == b"\xff\xd8\xff\xe0fake_jpeg_data"
        assert ts is not None

    def test_lidar_callback(self, bridge):
        msg = MagicMock()
        msg.ranges = [1.0, 2.0, float("inf"), 0.5]
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = math.pi / 2
        msg.range_min = 0.12
        msg.range_max = 3.5
        msg.header.stamp = self._make_stamp()

        bridge._lidar_cb(msg)
        scan = bridge.get_lidar()
        assert scan is not None
        assert scan["ranges"] == [1.0, 2.0, float("inf"), 0.5]
        assert scan["range_min"] == 0.12
        assert scan["range_max"] == 3.5

    def test_odom_callback(self, bridge):
        msg = MagicMock()
        msg.pose.pose.position.x = 1.0
        msg.pose.pose.position.y = 2.0
        msg.pose.pose.position.z = 0.0
        # Identity quaternion (yaw = 0)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.1
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        msg.header.stamp = self._make_stamp()

        bridge._odom_cb(msg)
        odom = bridge.get_odom()
        assert odom is not None
        assert abs(odom["position"]["x"] - 1.0) < 1e-6
        assert abs(odom["position"]["y"] - 2.0) < 1e-6
        assert abs(odom["orientation_yaw_deg"]) < 1.0  # ~0 degrees


# ---------------------------------------------------------------------------
# Motion: stop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop(self, bridge):
        bridge.stop()
        # Verify a Twist was published (publisher.publish was called)
        assert bridge._cmd_vel_pub.publish.called


# ---------------------------------------------------------------------------
# Motion: move_forward
# ---------------------------------------------------------------------------

class TestMoveForward:
    def _setup_odom_at(self, bridge, x=0.0, y=0.0, yaw_rad=0.0):
        """Seed the bridge with an odometry reading."""
        msg = MagicMock()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        s = math.sin(yaw_rad / 2)
        c = math.cos(yaw_rad / 2)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = s
        msg.pose.pose.orientation.w = c
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._odom_cb(msg)

    def test_no_odom_fails(self, bridge):
        result = bridge.move_forward(distance_m=0.5, speed=0.2, timeout=1.0)
        assert result["success"] is False
        assert "odometry" in result["message"].lower()

    def test_move_reaches_target(self, bridge):
        """Simulate odom updates during move to reach target."""
        self._setup_odom_at(bridge, x=0.0, y=0.0)

        # After the first check, simulate the robot having moved forward
        original_get_odom = bridge.get_odom
        call_count = [0]

        def mock_get_odom():
            call_count[0] += 1
            if call_count[0] >= 3:
                # Simulate robot has moved 0.5m in x
                return {
                    "position": {"x": 0.5, "y": 0.0, "z": 0.0},
                    "orientation_yaw_deg": 0.0,
                    "orientation_yaw_rad": 0.0,
                    "linear_velocity": {"x": 0.0, "y": 0.0},
                    "angular_velocity_z": 0.0,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                }
            return original_get_odom()

        bridge.get_odom = mock_get_odom
        result = bridge.move_forward(distance_m=0.5, speed=0.2, timeout=5.0)
        assert result["success"] is True
        assert result["distance_actual_m"] >= 0.5

    def test_speed_clamped(self, bridge):
        """Speed should be clamped to MAX_LINEAR_SPEED."""
        self._setup_odom_at(bridge, x=0.0, y=0.0)

        # Simulate having reached the target after a few odom reads
        call_count = [0]
        original_odom = bridge.get_odom()

        def mock_get_odom():
            call_count[0] += 1
            if call_count[0] >= 4:
                return {
                    "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "orientation_yaw_deg": 0.0,
                    "orientation_yaw_rad": 0.0,
                    "linear_velocity": {"x": 0.0, "y": 0.0},
                    "angular_velocity_z": 0.0,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                }
            return original_odom
        bridge.get_odom = mock_get_odom

        result = bridge.move_forward(distance_m=1.0, speed=5.0, timeout=5.0)
        # The function should have clamped internally — we just check it didn't crash
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Motion: turn
# ---------------------------------------------------------------------------

class TestTurn:
    def _setup_odom_at(self, bridge, x=0.0, y=0.0, yaw_rad=0.0):
        msg = MagicMock()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        s = math.sin(yaw_rad / 2)
        c = math.cos(yaw_rad / 2)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = s
        msg.pose.pose.orientation.w = c
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._odom_cb(msg)

    def test_no_odom_fails(self, bridge):
        result = bridge.turn(angle_deg=90, angular_speed=0.5, timeout=1.0)
        assert result["success"] is False

    def test_turn_reaches_target(self, bridge):
        self._setup_odom_at(bridge, yaw_rad=0.0)

        call_count = [0]
        def mock_get_odom():
            call_count[0] += 1
            if call_count[0] >= 3:
                return {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation_yaw_deg": 90.0,
                    "orientation_yaw_rad": math.pi / 2,
                    "linear_velocity": {"x": 0.0, "y": 0.0},
                    "angular_velocity_z": 0.0,
                    "timestamp": "2025-01-01T00:00:00+00:00",
                }
            return bridge.__class__.get_odom(bridge)

        bridge.get_odom = mock_get_odom
        result = bridge.turn(angle_deg=90.0, angular_speed=0.5, timeout=5.0)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# MCP tool wrappers (import after mocking)
# ---------------------------------------------------------------------------

class TestMCPTools:
    """Test the MCP tool functions directly (not via MCP transport)."""

    def test_get_camera_image_no_data(self, bridge):
        """get_camera_image should raise when no image available."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        with pytest.raises(ValueError, match="No camera image"):
            tb3_mcp.get_camera_image()

    def test_get_camera_image_with_data(self, bridge):
        """get_camera_image should return Image when data is available."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        # Simulate a camera callback
        msg = MagicMock()
        msg.data = b"\xff\xd8\xff\xe0fake_jpeg"
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._camera_cb(msg)

        result = tb3_mcp.get_camera_image()
        # FastMCP Image returns an Image object
        assert result is not None

    def test_get_lidar_scan_no_data(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.get_lidar_scan())
        assert "error" in result

    def test_get_odometry_no_data(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.get_odometry())
        assert "error" in result

    def test_stop_tool(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.stop())
        assert result["status"] == "stopped"

    def test_move_forward_negative_distance(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.move_forward(distance=-1.0))
        assert "error" in result

    def test_turn_zero_angle(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.turn(angle=0.0))
        assert "error" in result

    def test_diagnose_ros(self, bridge):
        """diagnose_ros should return diagnostic info about the bridge."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        result = json.loads(tb3_mcp.diagnose_ros())
        assert "environment" in result
        assert "bridge" in result
        assert result["bridge"]["camera_topic"] == "/camera/image/compressed"
        assert result["bridge"]["has_image"] is False
        assert result["bridge"]["has_odom"] is False
        assert "PID" in result["environment"]

    def test_open_camera_viewer(self, bridge):
        """open_camera_viewer should start the MJPEG server and return a URL."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        # Use a high ephemeral port to avoid conflicts
        result = json.loads(tb3_mcp.open_camera_viewer(port=0))
        # Port 0 will fail with OSError for real HTTPServer but our mock
        # environment may behave differently. However we can test the happy
        # path by using a real port.
        # For mock environment, just verify the function runs.
        assert "status" in result


# ---------------------------------------------------------------------------
# Camera viewer
# ---------------------------------------------------------------------------

class TestCameraViewer:
    """Test the MJPEG camera viewer in the bridge."""

    def test_start_viewer(self, bridge):
        """start_camera_viewer should start HTTP server and return URL."""
        # Use port 0 to let OS pick a free port
        url = bridge.start_camera_viewer(port=0, open_browser=False)
        assert url.startswith("http://localhost:")
        assert bridge.camera_viewer_url is not None
        assert bridge._viewer_server is not None
        assert bridge._viewer_thread.is_alive()
        bridge.stop_camera_viewer()
        assert bridge.camera_viewer_url is None

    def test_start_viewer_idempotent(self, bridge):
        """Starting the viewer twice should not create a second server."""
        url1 = bridge.start_camera_viewer(port=0, open_browser=False)
        url2 = bridge.start_camera_viewer(port=0, open_browser=False)
        assert url1 == url2
        bridge.stop_camera_viewer()

    def test_snapshot_endpoint(self, bridge):
        """The /snapshot endpoint should return a JPEG when image data exists."""
        import urllib.request

        url = bridge.start_camera_viewer(port=0, open_browser=False)
        port = bridge._viewer_port

        # Simulate camera callback with fake JPEG
        msg = MagicMock()
        msg.data = b"\xff\xd8\xff\xe0fake_jpeg_data"
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._camera_cb(msg)

        # Fetch snapshot
        resp = urllib.request.urlopen(f"http://localhost:{port}/snapshot")
        assert resp.status == 200
        data = resp.read()
        assert data == b"\xff\xd8\xff\xe0fake_jpeg_data"

        bridge.stop_camera_viewer()

    def test_html_page(self, bridge):
        """The / endpoint should return an HTML page."""
        import urllib.request

        url = bridge.start_camera_viewer(port=0, open_browser=False)
        port = bridge._viewer_port

        resp = urllib.request.urlopen(f"http://localhost:{port}/")
        assert resp.status == 200
        html = resp.read().decode()
        assert "TurtleBot3 Camera" in html

        bridge.stop_camera_viewer()


# ---------------------------------------------------------------------------
# chat.py image injection helper
# ---------------------------------------------------------------------------

class TestImageInjection:
    """Test _maybe_inject_image from chat.py."""

    def test_injects_image_for_valid_path(self, tmp_path):
        from src.model.serving.chat import _maybe_inject_image

        # Create a fake image file
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")

        messages = []
        _maybe_inject_image(str(img_path), messages)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_no_inject_for_text(self, tmp_path):
        from src.model.serving.chat import _maybe_inject_image

        messages = []
        _maybe_inject_image("Just some text response", messages)
        assert len(messages) == 0

    def test_no_inject_for_json(self, tmp_path):
        from src.model.serving.chat import _maybe_inject_image

        messages = []
        _maybe_inject_image('{"status": "ok"}', messages)
        assert len(messages) == 0

    def test_no_inject_for_nonexistent_file(self, tmp_path):
        from src.model.serving.chat import _maybe_inject_image

        messages = []
        _maybe_inject_image("/tmp/nonexistent_image_12345.png", messages)
        assert len(messages) == 0

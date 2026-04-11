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


class TestLidarDirectionalChecks:
    def _publish_scan(self, bridge, ranges, angle_min=-math.pi, angle_increment=math.pi / 180.0):
        msg = MagicMock()
        msg.ranges = ranges
        msg.angle_min = angle_min
        msg.angle_max = angle_min + angle_increment * len(ranges)
        msg.angle_increment = angle_increment
        msg.range_min = 0.12
        msg.range_max = 3.5
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._lidar_cb(msg)

    def test_check_obstacle_front_uses_angle_and_rotation(self, bridge):
        ranges = [float("inf")] * 360

        # With angle_min=-180° and LIDAR_FRAME_ROTATION_DEG=-90°,
        # physical front (0° in robot frame) maps to raw beam angle +90° => index 270.
        ranges[270] = 0.21
        self._publish_scan(bridge, ranges)

        front = bridge.check_obstacle(direction="front", arc_half_angle_deg=8)
        assert front is not None
        assert abs(front - 0.21) < 1e-6

    def test_check_obstacle_left_uses_angle_and_rotation(self, bridge):
        ranges = [float("inf")] * 360

        # With the same configuration, robot-left (90°) maps to raw -180° => index 0.
        ranges[0] = 0.34
        self._publish_scan(bridge, ranges)

        left = bridge.check_obstacle(direction="left", arc_half_angle_deg=8)
        assert left is not None
        assert abs(left - 0.34) < 1e-6


class TestFreshFrameWait:
    def _make_stamp(self, sec, nanosec=0):
        stamp = MagicMock()
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def _publish_frame(self, bridge, data, sec):
        msg = MagicMock()
        msg.data = data
        msg.header.stamp = self._make_stamp(sec)
        bridge._camera_cb(msg)

    def test_wait_for_fresh_frame_min_one(self, bridge):
        self._publish_frame(bridge, b"frame_0", 1000)

        def publish_next():
            time.sleep(0.02)
            self._publish_frame(bridge, b"frame_1", 1001)

        t = threading.Thread(target=publish_next)
        t.start()
        data, _ = bridge.wait_for_fresh_frame(
            timeout=0.3,
            min_new_frames=1,
            poll_interval=0.005,
        )
        t.join()

        assert data == b"frame_1"

    def test_wait_for_fresh_frame_min_two(self, bridge):
        self._publish_frame(bridge, b"frame_0", 1000)

        def publish_two():
            time.sleep(0.02)
            self._publish_frame(bridge, b"frame_1", 1001)
            time.sleep(0.02)
            self._publish_frame(bridge, b"frame_2", 1002)

        t = threading.Thread(target=publish_two)
        t.start()
        data, _ = bridge.wait_for_fresh_frame(
            timeout=0.5,
            min_new_frames=2,
            poll_interval=0.005,
        )
        t.join()

        assert data == b"frame_2"


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

    def test_get_lidar_scan_flags_likely_side_wall_stop(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        ranges = [float("inf")] * 360
        ranges[330] = 0.23
        ranges[0] = 0.66
        ranges[30] = 0.78

        msg = MagicMock()
        msg.ranges = ranges
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = math.pi / 180.0
        msg.range_min = 0.12
        msg.range_max = 3.5
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._lidar_cb(msg)

        result = json.loads(tb3_mcp.get_lidar_scan(summarize=True))
        assert "side_wall_risk" in result
        assert result["side_wall_risk"]["likely"] is True
        assert result["side_wall_risk"]["dominant_side"] == "right"

    def test_get_lidar_scan_no_side_wall_flag_when_center_is_nearest(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        ranges = [float("inf")] * 360
        ranges[330] = 0.52
        ranges[0] = 0.28
        ranges[30] = 0.49

        msg = MagicMock()
        msg.ranges = ranges
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = math.pi / 180.0
        msg.range_min = 0.12
        msg.range_max = 3.5
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._lidar_cb(msg)

        result = json.loads(tb3_mcp.get_lidar_scan(summarize=True))
        assert "side_wall_risk" in result
        assert result["side_wall_risk"]["likely"] is False

    def test_check_path_clear_front_includes_side_wall_warning(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        ranges = [float("inf")] * 360
        ranges[332] = 0.24
        ranges[0] = 0.70
        ranges[28] = 0.82

        msg = MagicMock()
        msg.ranges = ranges
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = math.pi / 180.0
        msg.range_min = 0.12
        msg.range_max = 3.5
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._lidar_cb(msg)

        result = json.loads(tb3_mcp.check_path_clear(direction="front", threshold_m=0.3))
        assert result["side_wall_risk"]["likely"] is True
        assert "warning" in result

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

    def test_move_forward_with_reactive_steer(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        bridge.move_forward = MagicMock(return_value={
            "success": True,
            "distance_requested_m": 0.5,
            "distance_actual_m": 0.5,
            "message": "ok",
        })

        result = json.loads(tb3_mcp.move_forward(
            distance=0.5,
            speed=0.2,
            reactive_steer=True,
            avoidance_angular_speed=0.3,
        ))

        assert result["success"] is True
        bridge.move_forward.assert_called_once_with(
            distance_m=0.5,
            speed=0.2,
            reactive_steer=True,
            avoidance_angular_speed=0.3,
        )

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

    def test_describe_scan_image(self, bridge):
        """describe_scan_image should forward annotations to the bridge."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        bridge.annotate_scan_frame = MagicMock(return_value={
            "ok": True,
            "scan_id": 1,
            "description": "soccer ball near center",
        })

        result = json.loads(tb3_mcp.describe_scan_image(description="soccer ball near center"))
        assert result["ok"] is True
        bridge.annotate_scan_frame.assert_called_once()


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


# ---------------------------------------------------------------------------
# Panoramic stitching helpers
# ---------------------------------------------------------------------------

def _make_test_jpeg(width=640, height=480, color=(128, 64, 32)):
    """Create a synthetic JPEG image as bytes."""
    from PIL import Image as PILImage
    import io
    img = PILImage.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_gradient_jpeg(width=640, height=480, offset=0):
    """Create a gradient JPEG (useful for feature-based stitching tests).

    The gradient shifts by *offset* pixels so adjacent frames share
    overlapping content — this gives OpenCV features to match on.
    """
    import numpy as np
    from PIL import Image as PILImage
    import io

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        val = int(((x + offset) % width) / width * 255)
        arr[:, x, :] = [val, 255 - val, 128]
    # Add some noise for feature matching
    noise = np.random.randint(0, 30, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class TestBuildGridMosaic:
    """Test the grid mosaic builder (fallback layout)."""

    def test_basic_grid(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_grid_mosaic
        from PIL import Image as PILImage
        import io

        frames = [_make_test_jpeg() for _ in range(6)]
        headings = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        result = _build_grid_mosaic(frames, headings)

        assert isinstance(result, bytes)
        assert len(result) > 100
        # Should be a valid JPEG
        img = PILImage.open(io.BytesIO(result))
        assert img.format == "JPEG"
        # Grid of 6 → 3×2
        assert img.width > 300
        assert img.height > 200

    def test_empty_frames_raises(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_grid_mosaic

        with pytest.raises(ValueError, match="No frames"):
            _build_grid_mosaic([], [])


class TestBuildContactSheet:
    """Test the annotated contact-sheet grid builder."""

    def test_basic_sheet(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_contact_sheet
        from PIL import Image as PILImage
        import io

        frames = [_make_test_jpeg(color=(i * 20, 100, 200)) for i in range(6)]
        headings = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
        result = _build_contact_sheet(frames, headings)

        assert isinstance(result, bytes)
        img = PILImage.open(io.BytesIO(result))
        assert img.format == "JPEG"
        # Grid should be reasonably sized (not an ultra-wide strip)
        assert img.width > 400
        assert img.height > 300

    def test_single_frame(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_contact_sheet

        frames = [_make_test_jpeg()]
        headings = [45.0]
        result = _build_contact_sheet(frames, headings)
        assert isinstance(result, bytes)
        assert len(result) > 100

    def test_empty_frames_raises(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_contact_sheet

        with pytest.raises(ValueError, match="No frames"):
            _build_contact_sheet([], [])

    def test_full_360_grid(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _build_contact_sheet
        from PIL import Image as PILImage
        import io

        # 15 frames at 24° steps covering 360°
        frames = [_make_test_jpeg(color=(i * 15, 50 + i * 10, 200)) for i in range(15)]
        headings = [i * 24.0 for i in range(15)]
        result = _build_contact_sheet(frames, headings)

        img = PILImage.open(io.BytesIO(result))
        # Grid has reasonable aspect ratio (not an extreme strip)
        ratio = img.width / max(img.height, 1)
        assert 0.5 < ratio < 4.0, f"Aspect ratio {ratio:.2f} is too extreme"


class TestStitchPanorama:
    """Test the hybrid _stitch_panorama function."""

    def test_returns_tuple(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _stitch_panorama

        frames = [_make_test_jpeg() for _ in range(4)]
        headings = [0.0, 30.0, 60.0, 90.0]
        result = _stitch_panorama(frames, headings)

        assert isinstance(result, tuple)
        assert len(result) == 2
        jpeg_bytes, method = result
        assert isinstance(jpeg_bytes, bytes)
        assert method in ('opencv_stitcher', 'contact_sheet')

    def test_method_label_is_string(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _stitch_panorama

        frames = [_make_test_jpeg() for _ in range(3)]
        headings = [0.0, 24.0, 48.0]
        _, method = _stitch_panorama(frames, headings)
        assert isinstance(method, str)
        assert len(method) > 0

    def test_empty_frames_raises(self):
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _stitch_panorama

        with pytest.raises(ValueError, match="No frames"):
            _stitch_panorama([], [])

    def test_fallback_without_opencv(self):
        """When cv2 is not importable, should fall back to contact_sheet."""
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _stitch_panorama
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'cv2':
                raise ImportError("Mocked cv2 not available")
            return original_import(name, *args, **kwargs)

        frames = [_make_test_jpeg(color=(i * 30, 100, 50)) for i in range(5)]
        headings = [0.0, 24.0, 48.0, 72.0, 96.0]

        with patch.object(builtins, '__import__', side_effect=mock_import):
            jpeg_bytes, method = _stitch_panorama(frames, headings)

        assert method == 'contact_sheet'
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 100

    def test_opencv_failure_falls_back(self):
        """When OpenCV stitcher fails, should fall back to contact_sheet."""
        from src.mcp.servers.tasks.robotics.turtlebot3.mcp_server import _stitch_panorama

        frames = [_make_test_jpeg() for _ in range(4)]
        headings = [0.0, 24.0, 48.0, 72.0]

        # Mock cv2 to return a failure status
        mock_cv2 = MagicMock()
        mock_cv2.Stitcher_PANORAMA = 0
        mock_cv2.Stitcher_OK = 0
        mock_stitcher = MagicMock()
        mock_stitcher.stitch.return_value = (1, None)  # status 1 = failure
        mock_cv2.Stitcher.create.return_value = mock_stitcher
        mock_cv2.imdecode = MagicMock(return_value=MagicMock())
        mock_cv2.IMREAD_COLOR = 1

        with patch.dict('sys.modules', {'cv2': mock_cv2}):
            jpeg_bytes, method = _stitch_panorama(frames, headings)

        assert method == 'contact_sheet'
        assert isinstance(jpeg_bytes, bytes)


class TestRotateAndScan:
    """Test the rotate_and_scan MCP tool."""

    def _setup_bridge_for_scan(self, bridge):
        """Configure bridge to deliver fake frames and odometry for a scan."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        # Seed odometry
        msg = MagicMock()
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._odom_cb(msg)

        # Make turn() always succeed
        step_count = [0]
        def mock_turn(angle_deg, angular_speed=0.5, timeout=30.0):
            step_count[0] += 1
            actual = abs(angle_deg)
            # Update odom heading
            new_yaw_deg = step_count[0] * actual
            new_yaw_rad = math.radians(new_yaw_deg)
            odom_msg = MagicMock()
            odom_msg.pose.pose.position.x = 0.0
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0
            s = math.sin(new_yaw_rad / 2)
            c = math.cos(new_yaw_rad / 2)
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = s
            odom_msg.pose.pose.orientation.w = c
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0
            odom_msg.header.stamp = stamp
            bridge._odom_cb(odom_msg)
            return {"success": True, "angle_actual_deg": actual}
        bridge.turn = mock_turn

        # Make wait_for_fresh_frame return a synthetic JPEG
        def mock_fresh_frame(timeout=1.5, min_new_frames=2, poll_interval=0.05):
            return _make_test_jpeg(color=(step_count[0] * 15, 100, 200)), "2025-01-01T00:00:00+00:00"
        bridge.wait_for_fresh_frame = mock_fresh_frame

        return tb3_mcp

    def test_rotate_and_scan_returns_panorama(self, bridge):
        """rotate_and_scan should return a summary and an image."""
        tb3_mcp = self._setup_bridge_for_scan(bridge)

        result = tb3_mcp.rotate_and_scan(total_angle=120.0, step_angle=30.0)

        assert isinstance(result, list)
        assert len(result) == 2
        summary = result[0]
        assert "Captured" in summary
        assert "Layout" in summary

    def test_rotate_and_scan_zero_frames_raises(self, bridge):
        """rotate_and_scan should raise when no frames are captured."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        # Seed odom
        msg = MagicMock()
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._odom_cb(msg)

        # Turn succeeds but no camera frames
        bridge.turn = MagicMock(return_value={"success": True, "angle_actual_deg": 24.0})
        bridge.wait_for_fresh_frame = MagicMock(return_value=(None, None))

        with pytest.raises(ValueError, match="captured 0 frames"):
            tb3_mcp.rotate_and_scan(total_angle=90.0, step_angle=30.0)

    def test_rotate_and_scan_default_step_is_24(self, bridge):
        """Default step_angle should be 24° (15 frames per 360°)."""
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        import inspect

        sig = inspect.signature(tb3_mcp.rotate_and_scan)
        assert sig.parameters["step_angle"].default == 24.0


class TestScanStep:
    def test_scan_step_returns_frame_and_summary(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        # Seed odometry
        msg = MagicMock()
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.angular.z = 0.0
        stamp = MagicMock()
        stamp.sec = 1000
        stamp.nanosec = 0
        msg.header.stamp = stamp
        bridge._odom_cb(msg)

        bridge.turn = MagicMock(return_value={"success": True, "angle_actual_deg": 12.0})
        bridge.wait_for_fresh_frame = MagicMock(
            return_value=(_make_test_jpeg(), "2025-01-01T00:00:00+00:00")
        )

        result = tb3_mcp.scan_step(step_angle=12.0)

        assert isinstance(result, list)
        assert len(result) == 2
        summary = json.loads(result[0])
        assert summary["success"] is True
        assert summary["frame"] == "ok"

    def test_scan_step_missing_frame_returns_json(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        bridge.turn = MagicMock(return_value={"success": True, "angle_actual_deg": 12.0})
        bridge.wait_for_fresh_frame = MagicMock(return_value=(None, None))

        result = tb3_mcp.scan_step(step_angle=12.0)
        payload = json.loads(result)

        assert payload["success"] is True
        assert payload["frame"] == "missing"


class TestScan360MVP:
    def test_scan_360_mvp_uses_frame_count(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        called = {}

        def _mock_rotate_and_scan(total_angle=360.0, step_angle=24.0, settle_time=0.2, speed=0.3):
            called["total_angle"] = total_angle
            called["step_angle"] = step_angle
            called["settle_time"] = settle_time
            called["speed"] = speed
            return ["ok", MagicMock()]

        with patch.object(tb3_mcp, "rotate_and_scan", side_effect=_mock_rotate_and_scan):
            result = tb3_mcp.scan_360_mvp(frame_count=20, settle_time=0.1, speed=0.4)

        assert isinstance(result, list)
        assert called["total_angle"] == 360.0
        assert abs(called["step_angle"] - 18.0) < 1e-9
        assert called["settle_time"] == 0.1
        assert called["speed"] == 0.4

    def test_scan_360_mvp_frame_count_limits(self, bridge):
        import src.mcp.servers.tasks.robotics.turtlebot3.mcp_server as tb3_mcp
        tb3_mcp._bridge = bridge

        with pytest.raises(ValueError, match="frame_count must be between 4 and 72"):
            tb3_mcp.scan_360_mvp(frame_count=3)

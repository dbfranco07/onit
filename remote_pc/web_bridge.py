#!/usr/bin/env python3
"""
Web Bridge Node — runs on Remote PC (Jetson Orin)
Combines:
  - ROS2 subscriber for camera images
  - ROS2 publisher for cmd_vel (continuous movement)
  - ROS2 service client for motion commands (single moves)
  - Flask + SocketIO web server serving the UI
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from tb3_interfaces.srv import MoveRobot

import cv2
import numpy as np
import base64
import threading
import os
import time

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO


class WebBridgeNode(Node):
    def __init__(self):
        super().__init__('web_bridge')

        # --- Camera Subscriber ---
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )
        self.latest_jpeg_b64 = None
        self.frame_count = 0

        # --- Direct cmd_vel Publisher (for continuous movement) ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.is_moving = False

        # Timer to continuously publish cmd_vel while moving
        self.cmd_vel_timer = self.create_timer(0.1, self.publish_cmd_vel)  # 10 Hz

        # --- Motion Service Client (for single-shot moves) ---
        self.motion_client = self.create_client(MoveRobot, '/move_robot')
        self.service_ready = False
        self.service_check_timer = self.create_timer(2.0, self.check_service)

        self.get_logger().info('Web bridge node initialized')

    def check_service(self):
        if not self.service_ready and self.motion_client.service_is_ready():
            self.service_ready = True
            self.get_logger().info('/move_robot service is available!')
            self.service_check_timer.cancel()

    def image_callback(self, msg):
        self.latest_jpeg_b64 = base64.b64encode(msg.data).decode('utf-8')
        self.frame_count += 1

    def publish_cmd_vel(self):
        """Continuously publish current velocity."""
        if self.is_moving:
            twist = Twist()
            twist.linear.x = self.current_linear
            twist.angular.z = self.current_angular
            self.cmd_vel_pub.publish(twist)

    def start_moving(self, linear_x, angular_z):
        """Start continuous movement."""
        self.current_linear = float(linear_x)
        self.current_angular = float(angular_z)
        self.is_moving = True
        self.get_logger().info(f'Moving: linear={linear_x}, angular={angular_z}')

    def stop_moving(self):
        """Stop the robot."""
        self.is_moving = False
        self.current_linear = 0.0
        self.current_angular = 0.0
        # Publish stop immediately
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info('Stopped')

    def send_motion_request(self, linear_x, angular_z, duration):
        """Send motion service request (for single-shot moves)."""
        if not self.service_ready:
            return None
        request = MoveRobot.Request()
        request.linear_x = float(linear_x)
        request.angular_z = float(angular_z)
        request.duration = float(duration)
        future = self.motion_client.call_async(request)
        return future


# ============================================================
# Flask + SocketIO Web Server
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIRS = [
    os.path.join(os.path.expanduser('~'), 'turtlebot3_ws', 'src', 'tb3_exercise', 'templates'),
    os.path.join(BASE_DIR, '..', 'templates'),
    os.path.join(BASE_DIR, 'templates'),
]

STATIC_DIRS = [
    os.path.join(os.path.expanduser('~'), 'turtlebot3_ws', 'src', 'tb3_exercise', 'static'),
    os.path.join(BASE_DIR, '..', 'static'),
    os.path.join(BASE_DIR, 'static'),
]

template_folder = None
static_folder = None
for d in TEMPLATE_DIRS:
    if os.path.isdir(d):
        template_folder = os.path.abspath(d)
        break
for d in STATIC_DIRS:
    if os.path.isdir(d):
        static_folder = os.path.abspath(d)
        break

if template_folder is None or static_folder is None:
    template_folder = '/tmp/tb3_templates'
    static_folder = '/tmp/tb3_static'
    os.makedirs(template_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)
    print(f'WARNING: Using fallback template dir: {template_folder}')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.config['SECRET_KEY'] = 'tb3_secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

ros_node = None


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('[WebSocket] Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('[WebSocket] Client disconnected')
    # Safety: stop robot if client disconnects
    if ros_node:
        ros_node.stop_moving()


@socketio.on('move_start')
def handle_move_start(data):
    """Start continuous movement when button is pressed."""
    global ros_node
    if ros_node is None:
        return

    linear_x = data.get('linear_x', 0.0)
    angular_z = data.get('angular_z', 0.0)

    ros_node.start_moving(linear_x, angular_z)
    socketio.emit('status', {
        'message': f'Moving: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}',
        'type': 'info'
    })


@socketio.on('move_stop')
def handle_move_stop(data=None):
    """Stop movement when button is released."""
    global ros_node
    if ros_node is None:
        return

    ros_node.stop_moving()
    socketio.emit('status', {
        'message': 'Stopped',
        'type': 'success'
    })


def stream_camera():
    """Background thread: emit camera frames to all connected WebSocket clients."""
    global ros_node
    while True:
        if ros_node is not None and ros_node.latest_jpeg_b64 is not None:
            socketio.emit('camera_frame', {
                'image': ros_node.latest_jpeg_b64
            })
        socketio.sleep(0.066)


def spin_ros(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def main(args=None):
    global ros_node

    rclpy.init(args=args)
    ros_node = WebBridgeNode()

    ros_thread = threading.Thread(target=spin_ros, args=(ros_node,), daemon=True)
    ros_thread.start()

    socketio.start_background_task(stream_camera)

    print('\n' + '=' * 50)
    print('  TurtleBot3 Web Control')
    print('  Open http://<THIS_IP>:5000 in your browser')
    print('=' * 50 + '\n')

    try:
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        if ros_node:
            ros_node.stop_moving()
            ros_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

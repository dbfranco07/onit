#!/usr/bin/env python3
"""
Video Publisher Node — runs on TurtleBot3
Subscribes to raw camera images from camera_ros,
compresses them as JPEG, and re-publishes as CompressedImage.

Optimisations for VLM agent comprehension:
  - Higher JPEG quality (default 80) for clearer object edges
  - Adaptive resize to consistent width (default 640) for bandwidth control
  - Optional CLAHE contrast enhancement for low-light lab environments
  - Configurable publish FPS to reduce bandwidth without dropping subscription
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
import time


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        # --- Configurable parameters ---
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('target_width', 640)
        self.declare_parameter('enhance_contrast', True)
        self.declare_parameter('publish_fps', 15.0)

        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.target_width = int(self.get_parameter('target_width').value)
        self.enhance_contrast = bool(self.get_parameter('enhance_contrast').value)
        self.publish_fps = float(self.get_parameter('publish_fps').value)

        self.jpeg_quality = max(40, min(95, self.jpeg_quality))
        self.target_width = max(0, self.target_width)
        self.publish_fps = max(1.0, min(30.0, self.publish_fps))

        # Frame-rate gating
        self._min_publish_interval = 1.0 / max(self.publish_fps, 1.0)
        self._last_publish_time = 0.0

        # CLAHE instance (reused across frames for efficiency)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Subscribe to raw images from camera_ros
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publish compressed images
        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/image/compressed',
            qos_profile_sensor_data
        )

        self.frame_received_count = 0
        self.frame_skipped_fps_count = 0
        self.frame_error_count = 0
        self.publish_count = 0
        self.get_logger().info(
            f'Video publisher started — quality={self.jpeg_quality}, '
            f'target_width={self.target_width}, enhance_contrast={self.enhance_contrast}, '
            f'publish_fps={self.publish_fps}'
        )

    def image_callback(self, msg):
        self.frame_received_count += 1

        # Frame-rate gating: skip if publishing too fast
        now = time.monotonic()
        if (now - self._last_publish_time) < self._min_publish_interval:
            self.frame_skipped_fps_count += 1
            return

        # Convert ROS Image to numpy array
        try:
            frame = self._decode_ros_image(msg)
        except Exception as e:
            self.frame_error_count += 1
            self.get_logger().warn(f'Failed to convert image: {e}')
            return

        # Rotate (camera is mounted sideways on the TurtleBot3)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Adaptive resize: scale to target_width while preserving aspect ratio
        h, w = frame.shape[:2]
        if self.target_width > 0 and w != self.target_width:
            scale = self.target_width / w
            new_w = self.target_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # CLAHE contrast enhancement (helps in low-light / uneven lab lighting)
        if self.enhance_contrast:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok, encoded = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            self.frame_error_count += 1
            self.get_logger().warn('Failed to encode JPEG frame')
            return

        # Publish compressed image
        comp_msg = CompressedImage()
        comp_msg.header = msg.header
        comp_msg.format = 'jpeg'
        comp_msg.data = encoded.tobytes()
        self.publisher.publish(comp_msg)

        self._last_publish_time = now
        self.publish_count += 1
        if self.publish_count % 30 == 0:
            self.get_logger().info(
                f'Published {self.publish_count} frames '
                f'(received {self.frame_received_count}, '
                f'skipped_fps {self.frame_skipped_fps_count}, '
                f'errors {self.frame_error_count}, {len(encoded)} bytes/frame)'
            )

    def _decode_ros_image(self, msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        if height <= 0 or width <= 0:
            raise ValueError(f'invalid image dimensions: {width}x{height}')

        data = np.frombuffer(msg.data, dtype=np.uint8)
        expected_rgb_bytes = height * width * 3

        if msg.encoding in ('bgr8', 'BGR888'):
            if data.size < expected_rgb_bytes:
                raise ValueError(
                    f'bgr8 frame too small: got {data.size}, need {expected_rgb_bytes}'
                )
            return data[:expected_rgb_bytes].reshape(height, width, 3)

        if msg.encoding in ('rgb8', 'RGB888'):
            if data.size < expected_rgb_bytes:
                raise ValueError(
                    f'rgb8 frame too small: got {data.size}, need {expected_rgb_bytes}'
                )
            rgb = data[:expected_rgb_bytes].reshape(height, width, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if msg.encoding in ('mono8', '8UC1'):
            expected_gray_bytes = height * width
            if data.size < expected_gray_bytes:
                raise ValueError(
                    f'mono8 frame too small: got {data.size}, need {expected_gray_bytes}'
                )
            gray = data[:expected_gray_bytes].reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        raise ValueError(f'Unsupported encoding: {msg.encoding}')


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

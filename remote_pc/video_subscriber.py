#!/usr/bin/env python3
"""
Video Subscriber Node — runs on Remote PC (Jetson Orin)
Subscribes to compressed camera images from TurtleBot3.
Can display via OpenCV window (standalone test) or be used by web_bridge.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('video_subscriber')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10
        )

        self.latest_frame = None
        self.frame_count = 0
        self.get_logger().info('Video subscriber started, waiting for images on /camera/image/compressed')

    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            self.latest_frame = frame
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(f'Received {self.frame_count} frames')

    def get_latest_frame(self):
        """Return the latest frame (used by web_bridge)."""
        return self.latest_frame


def main(args=None):
    """Standalone mode: shows camera feed in an OpenCV window."""
    rclpy.init(args=args)
    node = VideoSubscriber()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.latest_frame is not None:
                cv2.imshow('TurtleBot3 Camera', node.latest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
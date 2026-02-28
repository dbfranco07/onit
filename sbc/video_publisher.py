#!/usr/bin/env python3
"""
Video Publisher Node — runs on TurtleBot3
Subscribes to raw camera images from camera_ros,
compresses them as JPEG, and re-publishes as CompressedImage.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('jpeg_quality', 50)
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        # Subscribe to raw images from camera_ros
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish compressed images
        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/image/compressed',
            10
        )

        self.frame_count = 0
        self.get_logger().info('Video publisher started — subscribing to /camera/image_raw, publishing to /camera/image/compressed')

    def image_callback(self, msg):
        # Convert ROS Image to numpy array
        try:
            if msg.encoding == 'bgr8' or msg.encoding == 'BGR888':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'rgb8' or msg.encoding == 'RGB888':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Try to handle other formats via cv_bridge-style conversion
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        except Exception as e:
            self.get_logger().warn(f'Failed to convert image: {e}')
            return


        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, encoded = cv2.imencode('.jpg', frame, encode_params)

        # Publish compressed image
        comp_msg = CompressedImage()
        comp_msg.header = msg.header
        comp_msg.format = 'jpeg'
        comp_msg.data = encoded.tobytes()
        self.publisher.publish(comp_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Published {self.frame_count} compressed frames')


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

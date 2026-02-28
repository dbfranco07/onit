#!/usr/bin/env python3
"""
Motion Service Server — runs on TurtleBot3
Receives MoveRobot service requests and publishes cmd_vel commands.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tb3_interfaces.srv import MoveRobot
import time
import threading


class MotionServiceServer(Node):
    def __init__(self):
        super().__init__('motion_service_server')

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Service server
        self.srv = self.create_service(
            MoveRobot,
            '/move_robot',
            self.handle_move_request
        )

        # Safety limits
        self.max_linear = 0.22   # TurtleBot3 max linear speed (m/s)
        self.max_angular = 2.84  # TurtleBot3 max angular speed (rad/s)
        self.max_duration = 5.0  # Max duration per command (seconds)

        self.is_moving = False

        self.get_logger().info('Motion service server ready on /move_robot')

    def handle_move_request(self, request, response):
        self.get_logger().info(
            f'Received: linear_x={request.linear_x:.2f}, '
            f'angular_z={request.angular_z:.2f}, '
            f'duration={request.duration:.2f}s'
        )

        # Check if already executing a command
        if self.is_moving:
            response.success = False
            response.message = 'Robot is already executing a motion command'
            return response

        # Clamp values to safe limits
        linear_x = max(-self.max_linear, min(self.max_linear, request.linear_x))
        angular_z = max(-self.max_angular, min(self.max_angular, request.angular_z))
        duration = max(0.0, min(self.max_duration, request.duration))

        if duration <= 0.0:
            response.success = False
            response.message = 'Duration must be > 0'
            return response

        # Execute motion in a separate thread to avoid blocking
        self.is_moving = True
        thread = threading.Thread(
            target=self._execute_motion,
            args=(linear_x, angular_z, duration)
        )
        thread.start()
        thread.join()  # Wait for motion to complete before responding

        response.success = True
        response.message = (
            f'Executed: linear_x={linear_x:.2f} m/s, '
            f'angular_z={angular_z:.2f} rad/s for {duration:.2f}s'
        )
        return response

    def _execute_motion(self, linear_x, angular_z, duration):
        """Publish velocity for the specified duration, then stop."""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z

        rate = 10  # Hz
        steps = int(duration * rate)

        for _ in range(steps):
            self.cmd_vel_pub.publish(twist)
            time.sleep(1.0 / rate)

        # Stop the robot
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.is_moving = False
        self.get_logger().info('Motion complete — robot stopped')


def main(args=None):
    rclpy.init(args=args)
    node = MotionServiceServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure robot stops on shutdown
        stop_twist = Twist()
        node.cmd_vel_pub.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Motion Service Client — runs on Remote PC (Jetson Orin)
Sends MoveRobot service requests to TurtleBot3.
Standalone test mode: command-line interface.
"""

import rclpy
from rclpy.node import Node
from tb3_interfaces.srv import MoveRobot


class MotionServiceClient(Node):
    def __init__(self):
        super().__init__('motion_service_client')

        self.client = self.create_client(MoveRobot, '/move_robot')

        # Wait for service
        self.get_logger().info('Waiting for /move_robot service...')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Service not available, waiting...')

        self.get_logger().info('Motion service client connected!')

    def send_move_request(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """Send a motion request and return the result."""
        request = MoveRobot.Request()
        request.linear_x = float(linear_x)
        request.angular_z = float(angular_z)
        request.duration = float(duration)

        self.get_logger().info(
            f'Sending: linear_x={linear_x}, angular_z={angular_z}, duration={duration}'
        )

        future = self.client.call_async(request)
        return future


def main(args=None):
    """Standalone test: interactive command-line control."""
    rclpy.init(args=args)
    node = MotionServiceClient()

    print('\n=== TurtleBot3 Motion Client ===')
    print('Commands:')
    print('  f  - Move forward (0.1 m/s, 1s)')
    print('  b  - Move backward (-0.1 m/s, 1s)')
    print('  l  - Turn left (0.5 rad/s, 1s)')
    print('  r  - Turn right (-0.5 rad/s, 1s)')
    print('  q  - Quit')
    print('================================\n')

    commands = {
        'f': (0.1, 0.0, 1.0),
        'b': (-0.1, 0.0, 1.0),
        'l': (0.0, 0.5, 1.0),
        'r': (0.0, -0.5, 1.0),
    }

    try:
        while True:
            cmd = input('Enter command (f/b/l/r/q): ').strip().lower()
            if cmd == 'q':
                break
            if cmd not in commands:
                print('Unknown command')
                continue

            linear_x, angular_z, duration = commands[cmd]
            future = node.send_move_request(linear_x, angular_z, duration)

            # Spin until we get a response
            rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

            if future.result() is not None:
                result = future.result()
                print(f'  Result: success={result.success}, message="{result.message}"')
            else:
                print('  Service call failed or timed out')

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class DualServoPublisher(Node):
    def __init__(self):
        super().__init__('dual_servo_publisher')
        self.pub = self.create_publisher(String, 'servo_control', 10)
        self.get_logger().info("Dual Servo publisher ready.")

    def send_cmd(self, left, right):
        # Format like the ESP32 expects: L:x,R:y
        msg = String()
        msg.data = f"L:{left},R:{right}"
        self.pub.publish(msg)
        self.get_logger().info(f"Sent -> {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = DualServoPublisher()

    # If user passed values in terminal:
    # Example: python3 servo_control_publisher.py 90 255
    if len(sys.argv) == 3:
        left = sys.argv[1]
        right = sys.argv[2]
        node.send_cmd(left, right)
        rclpy.shutdown()
        return

    # Interactive CLI mode
    node.get_logger().info("Interactive mode. Enter L and R angles (0–180) or 255 to ignore.")

    try:
        while rclpy.ok():
            left = input("Left angle (0-180 or 255 ignore): ").strip()
            right = input("Right angle (0-180 or 255 ignore): ").strip()
            node.send_cmd(left, right)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()

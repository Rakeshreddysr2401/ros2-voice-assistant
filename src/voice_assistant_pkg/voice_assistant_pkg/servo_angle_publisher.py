#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16
import sys

class ServoPublisher(Node):
    def __init__(self):
        super().__init__('servo_angle_publisher')
        self.pub = self.create_publisher(UInt16, 'servo_angle', 10)
        self.get_logger().info("Servo publisher ready.")

    def send_angle(self, angle: int):
        msg = UInt16()
        msg.data = angle
        self.pub.publish(msg)
        self.get_logger().info(f"Sent angle: {angle}")

def main(args=None):
    rclpy.init(args=args)
    node = ServoPublisher()

    # If user passed angle in command line
    if len(sys.argv) > 1:
        angle = int(sys.argv[1])
        node.send_angle(angle)
        rclpy.shutdown()
        return

    # Interactive mode
    try:
        while rclpy.ok():
            angle = int(input("Enter angle (0–180): "))
            node.send_angle(angle)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()

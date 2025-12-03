#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import time


class OutputNode(Node):
    def __init__(self):
        super().__init__("output_node")

        self.subscription = self.create_subscription(
            String,
            "agent_response",
            self.handle_response,
            10
        )

        self.status_pub = self.create_publisher(String, "output_status", 10)

        self.get_logger().info("📢 OutputNode started (Console mode).")

    def handle_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        # Print the assistant response to the console
        print(f"\n🤖 Assistant says: {text}\n")

        # Simulate mic pause (optional)
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")

        # Simulate processing delay (optional)
        time.sleep(0.2)

        # Resume mic
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 0")

        # Notify input_node that "speaking" is done
        done = String()
        done.data = "speaking_done"
        self.status_pub.publish(done)


def main(args=None):
    rclpy.init(args=args)
    node = OutputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

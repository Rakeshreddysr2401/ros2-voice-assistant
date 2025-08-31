#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg
import threading
import time


class InputNode(Node):
    def __init__(self):
        super().__init__('input_node')

        # Publisher for user input
        self.input_publisher = self.create_publisher(
            String,
            'user_input',
            10
        )

        # Timer for periodic input check (can be replaced with voice input later)
        self.timer = self.create_timer(1.0, self.check_for_input)

        self.get_logger().info('Input Node started. Type messages and press Enter.')

        # Start input thread
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

        self.pending_input = None

    def input_loop(self):
        """Background thread for text input"""
        while rclpy.ok():
            try:
                user_input = input("Enter message: ")
                if user_input.strip():
                    self.pending_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                break

    def check_for_input(self):
        """Check for pending input and publish"""
        if self.pending_input:
            msg = VoiceAssistantMsg.create_input_msg(self.pending_input, "text")
            self.input_publisher.publish(msg)
            self.get_logger().info(f'Published input: {self.pending_input}')
            self.pending_input = None


def main(args=None):
    rclpy.init(args=args)
    node = InputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# output_node.py
# !/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyttsx3
import os
import time


class OutputNode(Node):
    """Text -> Speech using pyttsx3 TTS"""

    def __init__(self):
        super().__init__('output_node')
        self.subscription = self.create_subscription(
            String,
            'agent_response',
            self.handle_response,
            10
        )

        self.engine = pyttsx3.init()
        self.get_logger().info("🔊 OutputNode started (TTS enabled).")

    def handle_response(self, msg):
        # Simple string message - just get the text directly
        response_text = msg.data.strip()
        if not response_text:
            return

        print(f"\n🤖 Assistant: {response_text}\n")

        # Pause mic
        self.get_logger().info("🔇 Pausing mic while speaking...")
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")

        self.engine.say(response_text)
        self.engine.runAndWait()

        # Resume mic
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 0")
        time.sleep(0.2)
        self.get_logger().info("🎤 Mic resumed.")


def main(args=None):
    rclpy.init(args=args)
    node = OutputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
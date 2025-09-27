#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import time

PIPER_MODEL = "/home/rakhi/piper_models/en_US-amy-medium.onnx"


class OutputNode(Node):
    """Text -> Speech using Piper TTS (streamed to aplay)"""

    def __init__(self):
        super().__init__('output_node')
        self.subscription = self.create_subscription(
            String,
            'agent_response',
            self.handle_response,
            10
        )
        self.get_logger().info("🔊 OutputNode started (Piper TTS).")

    def handle_response(self, msg):
        response_text = msg.data.strip()
        if not response_text:
            return

        print(f"\n🤖 Assistant: {response_text}\n")

        # Pause mic while speaking
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")

        # Run Piper and stream raw audio directly to aplay
        piper = subprocess.Popen(
            ["piper", "--model", PIPER_MODEL, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        aplay = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=piper.stdout
        )
        # Send text to Piper
        piper.stdin.write(response_text.encode("utf-8"))
        piper.stdin.close()

        # Wait until playback finishes
        aplay.wait()

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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg
import pyttsx3
import os
import time

class OutputNode(Node):
    def __init__(self):
        super().__init__('output_node')
        self.subscription = self.create_subscription(
            String,                     # <-- use String directly
            'agent_response',
            self.handle_response,
            10
        )



        self.engine = pyttsx3.init()

        # Adjust properties
        voices = self.engine.getProperty('voices')
        # if voices:
        #     self.engine.setProperty('voice', voices[0].id)  # try voices[1], voices[2]... for variety
        # self.engine.setProperty('rate', 180)  # ~200 default, lower = clearer
        # self.engine.setProperty('volume', 1.0)  # max volume

        self.get_logger().info("🔊 OutputNode started (TTS enabled).")

    def handle_response(self, msg):
        data = VoiceAssistantMsg.parse_msg(msg)
        response_text = data['text']
        print(f"\n🤖 Assistant: {response_text}\n")

        # --- Tell input node to pause ---
        self.get_logger().info("🔇 Pausing mic while speaking...")
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 1")  # suspend mic

        self.engine.say(response_text)
        self.engine.runAndWait()

        # --- Resume mic after speaking ---
        os.system("pactl suspend-source @DEFAULT_SOURCE@ 0")  # resume mic
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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg
import pyttsx3

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
        self.get_logger().info("🔊 OutputNode started (TTS enabled).")

    def handle_response(self, msg):
        data = VoiceAssistantMsg.parse_msg(msg)
        response_text = data['text']
        print(f"\n🤖 Assistant: {response_text}\n")
        self.engine.say(response_text)
        self.engine.runAndWait()

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

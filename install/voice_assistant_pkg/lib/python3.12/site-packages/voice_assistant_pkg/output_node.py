#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg


class OutputNode(Node):
    def __init__(self):
        super().__init__('output_node')

        # Subscriber for agent responses
        self.response_subscription = self.create_subscription(
            String,
            'agent_response',
            self.handle_response,
            10
        )

        self.get_logger().info('Output Node started.')

    def handle_response(self, msg):
        """Handle incoming agent responses"""
        try:
            parsed_msg = VoiceAssistantMsg.parse_msg(msg)
            response_text = parsed_msg['text']

            # For now, just print (future: TTS output)
            print(f"\n🤖 Assistant: {response_text}\n")
            self.get_logger().info(f'Output: {response_text}')

            # Future: Add TTS functionality here
            # self.speak(response_text)

        except Exception as e:
            self.get_logger().error(f'Error handling response: {str(e)}')

    def speak(self, text: str):
        """Future TTS implementation"""
        # Placeholder for TTS functionality
        # You can use: pyttsx3, espeak, or festival
        pass


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
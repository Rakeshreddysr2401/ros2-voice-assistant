#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from voice_assistant_pkg.msg_types import VoiceAssistantMsg
import os
from typing import Optional

# Import LangChain components (install as needed)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_community.chat_models import ChatOpenAI
    from tavily import TavilyClient

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain components not available. Using simple echo mode.")


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # Subscriber for user input
        self.input_subscription = self.create_subscription(
            String,
            'user_input',
            self.process_input,
            10
        )

        # Publisher for agent response
        self.response_publisher = self.create_publisher(
            String,
            'agent_response',
            10
        )

        # Initialize LangChain components if available
        self.setup_agent()

        self.get_logger().info('Agent Node started.')

    def setup_agent(self):
        """Initialize the LangChain agent"""
        if not LANGCHAIN_AVAILABLE:
            self.get_logger().warn('LangChain not available. Using simple echo agent.')
            self.agent = None
            return

        # Setup your LLM and tools here
        # Example with OpenAI (you can replace with local models for Pi5)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
        else:
            self.get_logger().warn('No OpenAI API key found. Using echo mode.')
            self.llm = None

        # Setup Tavily for web search
        tavily_key = os.getenv('TAVILY_API_KEY')
        if tavily_key:
            self.tavily = TavilyClient(api_key=tavily_key)
        else:
            self.tavily = None
            self.get_logger().warn('No Tavily API key found.')

    def process_input(self, msg):
        """Process incoming user input"""
        try:
            parsed_msg = VoiceAssistantMsg.parse_msg(msg)
            user_text = parsed_msg['text']

            self.get_logger().info(f'Processing: {user_text}')

            # Process with agent
            response = self.generate_response(user_text)

            # Publish response
            response_msg = VoiceAssistantMsg.create_response_msg(response)
            self.response_publisher.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing input: {str(e)}')

    def generate_response(self, user_input: str) -> str:
        """Generate response using LangChain agent"""

        if not LANGCHAIN_AVAILABLE or not self.llm:
            # Simple echo response for testing
            return f"Echo: {user_input}"

        try:
            # Simple LLM call (you can expand this with LangGraph)
            messages = [
                SystemMessage(content="You are a helpful voice assistant running on a Raspberry Pi."),
                HumanMessage(content=user_input)
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            self.get_logger().error(f'Agent error: {str(e)}')
            return f"Sorry, I encountered an error: {str(e)}"


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
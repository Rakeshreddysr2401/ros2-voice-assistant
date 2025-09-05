# agent_node.py (Simplified)
# !/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Annotated, Literal
import operator
from langgraph.checkpoint.memory import MemorySaver


# ------------------ State Definition ------------------
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_input: str


# ------------------ Agent Node ------------------
class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # Simple in-memory storage
        self.memory = {}

        # ROS communication - simple String messages
        self.input_sub = self.create_subscription(String, 'user_input', self.process_input, 10)
        self.response_pub = self.create_publisher(String, 'agent_response', 10)

        # Service clients
        self.blip_client = self.create_client(Trigger, 'blip_describe')
        self.yolo_client = self.create_client(Trigger, 'yolo_detect')

        # LLM setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OPENAI_API_KEY not found in environment variables")
            self.llm = None
        else:
            self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)

        # Setup tools
        self.tools = self.setup_tools()
        self.tool_node = ToolNode(self.tools)

        # Create the graph
        self.memory_checkpointer = MemorySaver()
        self.graph = self.create_agent_graph()

        self.get_logger().info("🤖 AgentNode ready (simplified string messages)")

    def setup_tools(self):
        """Setup tools with actual service implementations"""

        @tool
        def get_visual_description(prompt: str = "") -> str:
            """Get a description of what the camera sees."""
            if not self.blip_client.service_is_ready():
                return "Camera service is currently unavailable"

            try:
                request = Trigger.Request()
                future = self.blip_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

                if future.result() and future.result().success:
                    return f"Camera sees: {future.result().message}"
                else:
                    return "Failed to get visual description from camera"
            except Exception as e:
                return f"Error getting visual description: {str(e)}"

        @tool
        def get_object_coordinates(object_name: str = "") -> str:
            """Detect objects and their coordinates using YOLO."""
            if not self.yolo_client.service_is_ready():
                return "Object detection service is currently unavailable"

            try:
                request = Trigger.Request()
                future = self.yolo_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

                if future.result() and future.result().success:
                    return f"Objects detected: {future.result().message}"
                else:
                    return "No objects were detected"
            except Exception as e:
                return f"Error during object detection: {str(e)}"

        @tool
        def store_memory(key: str, value: str) -> str:
            """Store information in memory."""
            try:
                self.memory[key] = value
                return f"Stored '{key}' in memory"
            except Exception as e:
                return f"Error storing in memory: {str(e)}"

        @tool
        def retrieve_memory(key: str) -> str:
            """Retrieve information from memory."""
            try:
                if key in self.memory:
                    return f"Retrieved: {key} = {self.memory[key]}"
                else:
                    available_keys = list(self.memory.keys())
                    if available_keys:
                        return f"Key '{key}' not found. Available keys: {available_keys}"
                    else:
                        return "Memory is empty"
            except Exception as e:
                return f"Error retrieving from memory: {str(e)}"

        return [get_visual_description, get_object_coordinates, store_memory, retrieve_memory]

    def create_agent_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.call_agent)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("respond", self.respond)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_use_tools,
            {
                "tools": "tools",
                "respond": "respond"
            }
        )

        # Add edges
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        return workflow.compile(checkpointer=self.memory_checkpointer)

    def call_agent(self, state: AgentState):
        """Main agent logic with LLM"""
        if not self.llm:
            return {
                "messages": [AIMessage(content="Language model is not available. Please check OpenAI API key.")]
            }

        messages = state.get("messages", [])
        user_input = state.get("user_input", "")

        # System message
        system_msg = SystemMessage(
            content="""You are a helpful AI assistant with access to vision and object detection tools. 
            You can:
            - See what the camera shows using get_visual_description
            - Detect objects and their positions using get_object_coordinates  
            - Store and retrieve information using memory tools

            Use these tools when appropriate. Be conversational and helpful."""
        )

        # Prepare messages
        conversation_messages = [system_msg] + messages + [HumanMessage(content=user_input)]

        try:
            # Bind tools to LLM and invoke
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(conversation_messages)

            return {"messages": [response]}

        except Exception as e:
            self.get_logger().error(f"Error in agent call: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Error processing request: {str(e)}")]
            }

    def should_use_tools(self, state: AgentState) -> Literal["tools", "respond"]:
        """Determine if we should use tools or respond directly"""
        last_message = state["messages"][-1] if state["messages"] else None

        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"

        return "respond"

    def respond(self, state: AgentState):
        """Generate final response and publish as simple string"""
        last_message = state["messages"][-1] if state["messages"] else None

        if last_message:
            response_text = last_message.content
        else:
            response_text = "I'm not sure how to respond to that."

        # Simple string message
        try:
            msg = String()
            msg.data = response_text
            self.response_pub.publish(msg)

            self.get_logger().info(f"Response: {response_text[:100]}...")

        except Exception as e:
            self.get_logger().error(f"Error publishing response: {str(e)}")

        return state

    def process_input(self, msg: String):
        """Process incoming user input - now just a simple string"""
        try:
            # Get the text directly from the string message
            user_text = msg.data.strip()

            if not user_text:
                self.get_logger().warning("Received empty user input")
                return

            self.get_logger().info(f"Processing: {user_text}")

            # Create initial state
            initial_state = {
                "messages": [],
                "user_input": user_text
            }

            # Thread ID for conversation continuity
            thread_id = {"configurable": {"thread_id": "main_conversation"}}

            # Invoke the graph
            self.graph.invoke(initial_state, config=thread_id)

        except Exception as e:
            self.get_logger().error(f"Error processing input: {str(e)}")
            # Simple error response
            error_msg = String()
            error_msg.data = f"Error: {str(e)}"
            self.response_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from typing import TypedDict, List, Annotated

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from .tools import get_tools, build_system_message  # dynamic tool loader

# ------------------ State Definition ------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

# ------------------ Agent Node ------------------
class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        # ROS communication
        self.input_sub = self.create_subscription(String, 'user_input', self.process_input, 10)
        self.response_pub = self.create_publisher(String, 'agent_response', 10)

        # Service clients (vision example)
        self.blip_client = self.create_client(Trigger, 'blip_describe')
        self.yolo_client = self.create_client(Trigger, 'yolo_detect')

        # LLM setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OPENAI_API_KEY not found in environment variables")
            self.llm = None
        else:
            self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)

        # Tools and graph
        if self.llm:
            self.tools = get_tools()
            self.tool_node = ToolNode(self.tools)
            self.memory_checkpointer = MemorySaver()
            self.graph = self.create_agent_graph()
        else:
            self.tools = []
            self.tool_node = None
            self.graph = None

        self.get_logger().info("🤖 AgentNode ready")

    def create_agent_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_agent)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END}
        )

        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.memory_checkpointer)

    def call_agent(self, state: AgentState):
        if not self.llm:
            return {"messages": [AIMessage(content="Language model is not available.")]}

        messages = state["messages"]

        # Dynamic system message from tools
        system_msg = build_system_message(self.tools)
        conversation_messages = [system_msg] + messages

        try:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(conversation_messages)

            return {"messages": [response] if isinstance(response, AIMessage) else [AIMessage(content=str(response))]}

        except Exception as e:
            self.get_logger().error(f"Error in agent call: {str(e)}")
            return {"messages": [AIMessage(content=f"Error processing request: {str(e)}")]}

    def process_input(self, msg: String):
        try:
            user_text = msg.data.strip()
            if not user_text:
                self.get_logger().warning("Received empty user input")
                return

            if not self.graph:
                error_msg = String()
                error_msg.data = "Language model not available. Please check OpenAI API key."
                self.response_pub.publish(error_msg)
                return

            self.get_logger().info(f"Processing: {user_text}")

            initial_state = {"messages": [HumanMessage(content=user_text)]}
            thread_id = {"configurable": {"thread_id": "main_conversation"}}

            result = self.graph.invoke(initial_state, config=thread_id)

            if result["messages"]:
                last_message = result["messages"][-1]
                response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)

                response_msg = String()
                response_msg.data = response_text
                self.response_pub.publish(response_msg)
                self.get_logger().info(f"Response: {response_text[:100]}...")
            else:
                self.get_logger().warning("No response generated")

        except Exception as e:
            self.get_logger().error(f"Error processing input: {str(e)}")
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

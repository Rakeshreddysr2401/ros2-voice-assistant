# agent_node.py
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from typing import TypedDict, List, Annotated

print("LangSmith key:", os.getenv("LANGSMITH_API_KEY"))
print("LangSmith project:", os.getenv("LANGSMITH_PROJECT"))

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from .tools import get_tools, build_system_message


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

        # LLM setup with better configuration
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OPENAI_API_KEY not found in environment variables")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000,  # Limit response length for faster processing
                timeout=30  # Add timeout for API calls
            )

        # Tools and graph setup
        if self.llm:
            self.tools = get_tools()
            self.tool_node = ToolNode(self.tools)
            self.memory_checkpointer = MemorySaver()
            self.graph = self.create_agent_graph()

            # Pre-build system message to avoid rebuilding it every time
            self.system_message = build_system_message(self.tools)
        else:
            self.tools = []
            self.tool_node = None
            self.graph = None
            self.system_message = None

        self.get_logger().info(f"🤖 AgentNode ready with {len(self.tools)} tools")

    def create_agent_graph(self):
        """Create the LangGraph workflow"""
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
        """Main agent logic with improved error handling"""
        if not self.llm:
            return {"messages": [AIMessage(content="Language model is not available. Please check OpenAI API key.")]}

        try:
            messages = state["messages"]

            # Ensure we have a system message at the beginning
            if not messages or not isinstance(messages[0], SystemMessage):
                # Insert system message at the beginning
                conversation_messages = [self.system_message] + messages
            else:
                conversation_messages = messages

            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(self.tools)

            # Get response from LLM
            response = llm_with_tools.invoke(conversation_messages)

            # Log tool usage for debugging
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tool_call.get('name', 'unknown') for tool_call in response.tool_calls]
                self.get_logger().info(f"LLM requested tools: {tool_names}")

            return {"messages": [response]}

        except Exception as e:
            self.get_logger().error(f"Error in agent call: {str(e)}")
            error_response = AIMessage(
                content=f"I encountered an error while processing your request: {str(e)}. "
                        "Please try rephrasing your question or check if all services are running."
            )
            return {"messages": [error_response]}

    def process_input(self, msg: String):
        """Process incoming user input with improved error handling and logging"""
        try:
            user_text = msg.data.strip()
            if not user_text:
                self.get_logger().warning("Received empty user input")
                return

            if not self.graph:
                error_msg = String()
                error_msg.data = "I'm not ready yet. Please check that the OpenAI API key is configured correctly."
                self.response_pub.publish(error_msg)
                return

            self.get_logger().info(f"Processing user input: {user_text}")

            # Create initial state with user message
            initial_state = {"messages": [HumanMessage(content=user_text)]}

            # Use consistent thread ID for conversation memory
            thread_config = {"configurable": {"thread_id": "main_conversation"}}

            # Process with timeout
            result = self.graph.invoke(initial_state, config=thread_config)

            # Extract and publish response
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]

                if hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = str(last_message)

                # Ensure response is not empty
                if not response_text.strip():
                    response_text = "I processed your request but didn't generate a response. Could you please rephrase your question?"

                response_msg = String()
                response_msg.data = response_text
                self.response_pub.publish(response_msg)

                # Log response (truncated for readability)
                log_text = response_text[:150] + "..." if len(response_text) > 150 else response_text
                self.get_logger().info(f"Response generated: {log_text}")
            else:
                self.get_logger().warning("No response generated from agent")
                error_msg = String()
                error_msg.data = "I'm having trouble generating a response. Please try again."
                self.response_pub.publish(error_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing input: {str(e)}")
            error_msg = String()
            error_msg.data = f"I encountered an error: {str(e)}. Please try again or check if all services are running."
            self.response_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down AgentNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
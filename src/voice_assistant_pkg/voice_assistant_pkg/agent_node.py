# agent_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from .tools import get_tools
from .states.states import AgentState
from .agents.chatAgentNode import call_agent
from .llm_config import llm

MAX_RETRIES = 2


# ------------------ Agent Node ------------------
class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.llm = llm
        # ROS communication
        self.input_sub = self.create_subscription(String, 'user_input', self.process_input, 10)
        self.response_pub = self.create_publisher(String, 'agent_response', 10)

        # Tools and graph setup
        if self.llm:
            self.tools = get_tools()
            self.tool_node = ToolNode(self.tools)
            self.graph = self.create_agent_graph()
        else:
            self.tools = []
            self.tool_node = None
            self.graph = None
        self.get_logger().info(f"🤖 AgentNode ready with {len(self.tools)} tools")

    def create_agent_graph(self):

        def chat_agent_transition(state: AgentState):
            """Determine next step after chatAgent."""
            messages = state.get("messages", [])

            if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
                return "tools"
            return END

        def reviewer_transition(state):
            """Determine next step after reviewerAgent."""
            feedback = state.get("review_feedback", {})
            retry_count = state.get("retry_count", 0)
            satisfied = feedback.get("satisfied", True)

            if satisfied or retry_count > MAX_RETRIES:
                return END
            elif not satisfied and retry_count <= MAX_RETRIES:
                return "agent"
            return END

        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_agent)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            chat_agent_transition,
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=MemorySaver())

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
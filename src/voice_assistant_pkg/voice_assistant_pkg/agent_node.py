import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .states.states import AgentState
from .agents.chat_agent import call_parent_agent
from .llm_config import llm  # just to check it's configured


class AgentNode(Node):
    def __init__(self):
        super().__init__("agent_node")

        self.llm_ready = llm is not None

        # ROS I/O
        self.input_sub = self.create_subscription(
            String, "user_input", self.process_input, 10
        )
        self.response_pub = self.create_publisher(String, "agent_response", 10)

        # LangGraph
        if self.llm_ready:
            self.graph = self._build_graph()
        else:
            self.graph = None

        self.get_logger().info("🤖 Multi-Agent AgentNode initialised.")

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("chat_parent", call_parent_agent)
        workflow.set_entry_point("chat_parent")

        # No extra edges needed: parent agent uses tool-calling
        return workflow.compile(checkpointer=MemorySaver())

    def process_input(self, msg: String):
        try:
            user_text = msg.data.strip()
            if not user_text:
                self.get_logger().warning("Empty user input received.")
                return

            if not self.llm_ready or self.graph is None:
                out = String()
                out.data = (
                    "I'm not ready yet. Please check the LLM/API configuration."
                )
                self.response_pub.publish(out)
                return

            self.get_logger().info(f"User: {user_text}")

            initial_state = {"messages": [HumanMessage(content=user_text)]}
            thread_config = {"configurable": {"thread_id": "main_conversation"}}

            result = self.graph.invoke(initial_state, config=thread_config)

            messages = result.get("messages", [])
            if not messages:
                response_text = (
                    "I processed your request but didn't generate a response."
                )
            else:
                last = messages[-1]
                response_text = getattr(last, "content", str(last))

            out = String()
            out.data = response_text
            self.response_pub.publish(out)

            log_text = (
                response_text[:150] + "..."
                if len(response_text) > 150
                else response_text
            )
            self.get_logger().info(f"Response: {log_text}")

        except Exception as e:
            self.get_logger().error(f"Error in AgentNode: {e}")
            out = String()
            out.data = (
                "I encountered an error while processing your request. "
                f"Details: {e}"
            )
            self.response_pub.publish(out)


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

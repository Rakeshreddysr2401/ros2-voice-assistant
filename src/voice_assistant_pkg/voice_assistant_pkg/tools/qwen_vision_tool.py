# -------------------------------------------------------------
# tools/qwen_vision_tool.py
# -------------------------------------------------------------
# A LangGraph / LangChain tool that talks to qwen_vision_server.
# - describe_scene(): calls the /qwen_vision_describe Trigger service and returns the cached description
# - ask_scene(question: str): publishes the question to /vision_query and waits for /vision_response


import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String
from langchain_core.tools import tool
import threading

# Globals to hold a single node across tool calls
_rcl_inited = False
_node: Node | None = None


def _ensure_node():
    global _rcl_inited, _node
    if not _rcl_inited:
        if not rclpy.ok():
            rclpy.init()
        _node = Node('qwen_vision_tool_client')
        _rcl_inited = True
    return _node


@tool
def describe_scene() -> str:
    """Get a cached description (and prime the model if needed) from the qwen vision server."""
    node = _ensure_node()
    client = node.create_client(Trigger, 'qwen_vision_describe')
    if not client.wait_for_service(timeout_sec=15.0):
        return 'Qwen vision describe service not available.'

    req = Trigger.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)

    if future.done() and future.result() is not None:
        res = future.result()
        return res.message if res.success else f'Failed to get description: {res.message}'
    else:
        return 'Timed out waiting for qwen_vision_describe service.'


@tool
def ask_scene(question: str) -> str:
    """
    Ask a follow-up question about the most recently primed image. The tool publishes the question to
    /vision_query and waits for a single /vision_response message.
    """
    node = _ensure_node()

    # Prepare a container to receive the response
    response_container = {'text': None}
    response_event = threading.Event()

    def _on_response(msg: String):
        response_container['text'] = msg.data
        response_event.set()

    sub = node.create_subscription(String, 'vision_response', _on_response, 10)

    pub = node.create_publisher(String, 'vision_query', 10)

    # Publish the question
    qmsg = String()
    qmsg.data = question
    pub.publish(qmsg)

    # Wait up to 20s for an answer
    waited = response_event.wait(timeout=20.0)

    # Clean up subscriptions/publishers (ROS2 Python will garbage collect but explicit removal is fine)
    node.destroy_subscription(sub)

    if not waited:
        return 'Timed out waiting for vision response.'
    return response_container['text']
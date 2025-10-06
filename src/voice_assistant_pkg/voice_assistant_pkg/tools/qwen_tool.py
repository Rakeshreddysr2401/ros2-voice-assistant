import rclpy
from rclpy.node import Node
from custom_interfaces.srv import QwenVision
from langchain_core.tools import tool

_rcl_inited = False
_node: Node | None = None

def _ensure_node():
    global _rcl_inited, _node
    if not _rcl_inited:
        if not rclpy.ok():
            rclpy.init()
        _node = Node('qwen_tool_client')
        _rcl_inited = True
    return _node

@tool
def qwen_vision_tool(query: str, use_latest: bool = True) -> str:
    """
    Ask the Qwen Vision server to describe an image.
    Args:
        query: The text query for the image model.
        use_latest: If True, uses the most recent frame. If False, reuses the previous captured frame.
    """

    print(f"🤖 Searching Qwen Vision Tool with Query: {query} \n And using Latest Frame: {use_latest}\n")
    node = _ensure_node()
    client = node.create_client(QwenVision, 'qwen_vision_describe')
    if not client.wait_for_service(timeout_sec=30.0):
        return "QwenVision service not available."

    req = QwenVision.Request()
    req.query = query
    req.use_latest = use_latest

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=75.0)

    if future.done() and future.result() is not None:
        res = future.result()
        return res.description if res.success else f"Failed: {res.description}"
    else:
        return "Timed out waiting for QwenVision service."

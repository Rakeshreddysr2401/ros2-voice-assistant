import rclpy
from rclpy.node import Node
from custom_interfaces.srv import QwenVision # Reusing the vision service interface
from langchain_core.tools import tool

# Global ROS node for the tool
_rcl_inited = False
_node: Node | None = None

def _ensure_node():
    global _rcl_inited, _node
    if not _rcl_inited:
        if not rclpy.ok():
            rclpy.init()
        _node = Node('moondream_tool_client')
        _rcl_inited = True
    return _node

@tool
def fast_vision_tool(query: str) -> str:
    """
    Use this for FAST visual checks (responses in <1s). 
    Perfect for tracking objects, checking if a path is clear, or confirming an object's presence while moving.
    Example: 'is the bottle still in the center?', 'is there an obstacle in front?'
    """
    node = _ensure_node()
    # We will point this to a specialized fast server instance
    client = node.create_client(QwenVision, 'moondream_vision_describe')
    
    if not client.wait_for_service(timeout_sec=5.0):
        return "Fast vision service (Moondream) not available. Make sure the moondream_server is running."

    req = QwenVision.Request()
    req.query = query
    req.use_latest = True

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

    if future.done() and future.result() is not None:
        return future.result().response
    else:
        return "Timed out waiting for Moondream response."

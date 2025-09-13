import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from langchain_core.tools import tool

# Create a global node just once
_rcl_inited = False
_node: Node | None = None

def _ensure_node():
    global _rcl_inited, _node
    if not _rcl_inited:
        if not rclpy.ok():
            rclpy.init()
        _node = Node('blip_tool_client')
        _rcl_inited = True
    return _node

@tool
def describe_scene() -> str:
    """
    Capture the current camera frame and return a description using the BLIP model.
    """
    node = _ensure_node()
    client = node.create_client(Trigger, 'blip_describe')
    if not client.wait_for_service(timeout_sec=10.0):
        return "BLIP service not available."

    req = Trigger.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)

    if future.done() and future.result() is not None:
        res = future.result()
        return res.message if res.success else f"Failed to get description: {res.message}"
    else:
        return "Timed out waiting for BLIP description service."

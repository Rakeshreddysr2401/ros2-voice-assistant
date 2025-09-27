#tools.yolo_tool.py
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
        _node = Node('yolo_tool_client')
        _rcl_inited = True
    return _node

@tool
def describe_objects() -> str:
    """
    Detect and locate specific objects in the current camera view using YOLO object detection.
    Use this when the user asks about specific objects, their locations, positions, or coordinates.
    Returns detailed information about detected objects including their positions and confidence scores.
    """
    node = _ensure_node()
    client = node.create_client(Trigger, 'yolo_detect')
    if not client.wait_for_service(timeout_sec=15.0):
        return "YOLO service not available."

    req = Trigger.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)

    if future.done() and future.result() is not None:
        res = future.result()
        return res.message if res.success else f"Failed to get object coordinates: {res.message}"
    else:
        return "Timed out waiting for YOLO service."

#voice_assistant_pkg.tools.yolo_tool.py
import rclpy
from rclpy.node import Node
from custom_interfaces.srv import YoloDetect
from langchain_core.tools import tool

# Global ROS node
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
def describe_objects(use_latest: bool = True) -> str:
    """
    Detect objects using YOLO.
    - use_latest=True: uses latest frame (and updates first_frame)
    - use_latest=False: uses first_frame
    """
    node = _ensure_node()
    client = node.create_client(YoloDetect, 'yolo_detect')
    if not client.wait_for_service(timeout_sec=15.0):
        return "YOLO service not available."

    req = YoloDetect.Request()
    req.use_latest = use_latest

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)

    if future.done() and future.result() is not None:
        res = future.result()
        return res.message if res.success else f"Failed to get object coordinates: {res.message}"
    else:
        return "Timed out waiting for YOLO service."

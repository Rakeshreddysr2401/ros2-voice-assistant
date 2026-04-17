import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_core.tools import tool

_node = None

def _ensure_node():
    global _node
    if _node is None:
        if not rclpy.ok():
            rclpy.init()
        _node = Node('tracking_tool_client')
        _node.pub = _node.create_publisher(String, 'servoing_control', 10)
    return _node

@tool
def track_object(target_label: str, action: str = "START") -> str:
    """
    Control active visual tracking (visual servoing).
    target_label: The object to track (e.g., 'bottle').
    action: 'START' to begin tracking/following, 'STOP' to end it.
    Use this for smooth, continuous movement toward an object.
    """
    node = _ensure_node()
    msg = String()
    
    action = action.upper()
    if action == "START":
        msg.data = f"START:{target_label}"
        node.pub.publish(msg)
        return f"🚀 Started active visual servoing to reach the {target_label}."
    else:
        msg.data = "STOP"
        node.pub.publish(msg)
        return "🛑 Visual servoing stopped."

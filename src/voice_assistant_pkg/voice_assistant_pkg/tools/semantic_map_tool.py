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
        _node = Node('semantic_map_tool_client')
        _node.pub_pin = _node.create_publisher(String, 'pin_object', 10)
        _node.pub_query = _node.create_publisher(String, 'query_map', 10)
    return _node

@tool
def pin_object_on_map(label: str) -> str:
    """
    Mark the current location of an object on the robot's internal semantic map.
    Use this when you have successfully arrived at an object or identified its location clearly.
    """
    node = _ensure_node()
    msg = String()
    msg.data = label
    node.pub_pin.publish(msg)
    return f"📍 Pinned {label} at my current location on the semantic map."

@tool
def query_semantic_map() -> str:
    """
    Ask the robot for a summary of all known object locations relative to its current position.
    Use this when the user asks 'where is the [object]?' or to plan a path to a previously seen item.
    """
    node = _ensure_node()
    msg = String()
    msg.data = "GET"
    node.pub_query.publish(msg)
    return "🔍 Querying semantic map... check the dashboard or agent status for the result."

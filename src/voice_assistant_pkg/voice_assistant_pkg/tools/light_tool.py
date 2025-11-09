from langchain_core.tools import tool
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

_node = None


def get_node():
    global _node
    if _node is None:
        if not rclpy.ok():
            rclpy.init()
        _node = Node("traffic_light_tool_node")
    return _node


@tool
def set_traffic_light(color: str):
    """Send a traffic light command: RED, GREEN, or ORANGE."""
    node = get_node()
    pub = node.create_publisher(String, "traffic_light_cmd", 10)
    msg = String()
    msg.data = color.upper()

    pub.publish(msg)
    return f"Traffic light set to {color}"

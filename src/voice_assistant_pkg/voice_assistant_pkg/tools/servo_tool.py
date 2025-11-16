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
        _node = Node("servo_tool_node")
        _node.pub = _node.create_publisher(String, "servo_control", 10)
    return _node

def _format_cmd(left, right):
    l = 255 if left is None else max(0, min(180, int(left)))
    r = 255 if right is None else max(0, min(180, int(right)))
    return f"L:{l},R:{r}"

@tool
def move_servos(left: int | None = None, right: int | None = None):
    """
    Control the robot's left and right servos.

    - left: 0-180 or None for ignore
    - right: 0-180 or None for ignore
    """
    node = _ensure_node()
    cmd = _format_cmd(left, right)
    msg = String()
    msg.data = cmd
    node.pub.publish(msg)
    return f"Sent servo command: {cmd}"

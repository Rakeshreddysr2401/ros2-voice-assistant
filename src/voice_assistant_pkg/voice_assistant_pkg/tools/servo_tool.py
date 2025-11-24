# tools.servo_tool.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt16
from langchain_core.tools import tool

# Global ROS node & publisher holder
_rcl_inited = False
_node = None
_pub = None


def _ensure_servo_node():
    """Create one global ROS node + publisher for servo_angle."""
    global _rcl_inited, _node, _pub

    if not _rcl_inited:
        if not rclpy.ok():
            rclpy.init()

        _node = Node("servo_tool_node")

        # Publisher for a single servo
        _pub = _node.create_publisher(UInt16, "servo_angle", 10)

        _node.get_logger().info("🔥 Servo tool node initialized!")
        _rcl_inited = True

    return _node, _pub


@tool
def move_servos(angle: int):
    """
    Move the servo to the given angle (0–180).
    Example: move_servos(90)
    """
    node, pub = _ensure_servo_node()

    # Safety clamp
    angle = max(0, min(180, angle))

    msg = UInt16()
    msg.data = angle
    pub.publish(msg)

    node.get_logger().info(f"🔧 Published servo angle: {angle}")

    return f"Servo moved to {angle}°"

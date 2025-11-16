import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_core.tools import tool
from typing import Optional, Literal

_node = None


def _ensure_node():
    """Ensure ROS node is initialized for servo control"""
    global _node
    if _node is None:
        if not rclpy.ok():
            rclpy.init()
        _node = Node("servo_tool_node")
        _node.pub = _node.create_publisher(String, "servo_control", 10)
    return _node


def _format_cmd(left: Optional[int], right: Optional[int]) -> str:
    """
    Format servo command string.
    255 indicates 'no change' for that servo.
    """
    l = 255 if left is None else max(0, min(180, int(left)))
    r = 255 if right is None else max(0, min(180, int(right)))
    return f"L:{l},R:{r}"


@tool
def move_servos(
        left: Optional[int] = None,
        right: Optional[int] = None,
        action: Optional[str] = None
) -> str:
    """
    Control the humanoid robot's arm servos (left and right hands).

    Args:
        left: Left arm angle (0-180 degrees). 0=down, 90=horizontal, 180=up. None=no change.
        right: Right arm angle (0-180 degrees). 0=down, 90=horizontal, 180=up. None=no change.
        action: Optional preset action like "wave", "raise_both", "lower_both", "rest"

    Examples:
        - Raise left arm: left=180
        - Lower right arm: right=0
        - Both arms horizontal: left=90, right=90
        - Wave: action="wave"
        - Rest position: action="rest"

    Returns:
        Confirmation message of the servo command sent.
    """
    # Handle preset actions
    if action:
        action_lower = action.lower()
        if action_lower == "wave":
            # Wave sequence could be handled by multiple calls
            left, right = 90, 180
        elif action_lower == "raise_both":
            left, right = 180, 180
        elif action_lower == "lower_both":
            left, right = 0, 0
        elif action_lower == "rest":
            left, right = 45, 45
        elif action_lower == "horizontal":
            left, right = 90, 90

    # Validate inputs
    if left is not None and not (0 <= left <= 180):
        return f"Error: left arm angle must be 0-180, got {left}"
    if right is not None and not (0 <= right <= 180):
        return f"Error: right arm angle must be 0-180, got {right}"

    if left is None and right is None:
        return "Error: Must specify at least one arm position or action"

    # Send command
    node = _ensure_node()
    cmd = _format_cmd(left, right)
    msg = String()
    msg.data = cmd
    node.pub.publish(msg)

    # Build response
    response_parts = []
    if left is not None:
        response_parts.append(f"left arm to {left}°")
    if right is not None:
        response_parts.append(f"right arm to {right}°")

    action_desc = f" ({action})" if action else ""
    return f"✅ Moved {' and '.join(response_parts)}{action_desc}"
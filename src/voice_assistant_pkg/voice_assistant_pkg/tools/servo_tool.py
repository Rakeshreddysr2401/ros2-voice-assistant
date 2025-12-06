#voice_assistant_pkg.tools.servo_tool.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from langchain_core.tools import tool
from typing import Optional

_node = None

def _ensure_node():
    """Ensure ROS node is initialized for servo control"""
    global _node
    if _node is None:
        print("[DEBUG] Initializing servo_tool_node...")
        if not rclpy.ok():
            print("[DEBUG] rclpy NOT initialized. Calling rclpy.init()...")
            rclpy.init()
        _node = Node("servo_tool_node")
        _node.pub = _node.create_publisher(String, "servo_control", 10)
        print("[DEBUG] Publisher initialized on topic: servo_control")
    else:
        print("[DEBUG] servo_tool_node already exists")
    return _node


def _format_cmd(left: Optional[int], right: Optional[int]) -> str:
    l = 255 if left is None else max(0, min(180, int(left)))
    r = 255 if right is None else max(0, min(180, int(right)))
    cmd = f"L:{l},R:{r}"
    print(f"[DEBUG] Formatted CMD = {cmd}")
    return cmd


@tool
def move_servos(left: Optional[int] = None,
                right: Optional[int] = None,
                action: Optional[str] = None) -> str:
    """Control the humanoid robot's arm servos with debug logs."""

    print(f"[DEBUG] move_servos called → left={left}, right={right}, action={action}")

    # Handle preset actions
    if action:
        action_lower = action.lower()
        print(f"[DEBUG] Processing action preset: {action_lower}")

        if action_lower == "wave":
            left, right = 90, 180
        elif action_lower == "raise_both":
            left, right = 180, 180
        elif action_lower == "lower_both":
            left, right = 0, 0
        elif action_lower == "rest":
            left, right = 45, 45
        elif action_lower == "horizontal":
            left, right = 90, 90

    # Validate range
    if left is not None and not (0 <= left <= 180):
        print(f"[ERROR] Invalid left angle: {left}")
        return f"Error: left arm angle must be 0-180, got {left}"

    if right is not None and not (0 <= right <= 180):
        print(f"[ERROR] Invalid right angle: {right}")
        return f"Error: right arm angle must be 0-180, got {right}"

    if left is None and right is None:
        print("[ERROR] No servo values specified")
        return "Error: Must specify at least one arm position or action"

    # Publish to ROS topic
    node = _ensure_node()
    cmd = _format_cmd(left, right)

    msg = String()
    msg.data = cmd

    print(f"[DEBUG] Publishing to /servo_control → {cmd}")
    node.pub.publish(msg)

    # Build response text
    response_parts = []
    if left is not None:
        response_parts.append(f"left arm to {left}°")
    if right is not None:
        response_parts.append(f"right arm to {right}°")

    action_desc = f" ({action})" if action else ""
    final_msg = f"✅ Moved {' and '.join(response_parts)}{action_desc}"

    print(f"[DEBUG] move_servos returning: {final_msg}")

    return final_msg

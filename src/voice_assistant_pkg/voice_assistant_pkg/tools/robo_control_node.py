import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt16
from langchain_core.tools import tool
from typing import Optional

_node = None

# -----------------------------------------
# NODE INITIALIZER (Shared for both tools)
# -----------------------------------------
def _ensure_node():
    """Ensure ROS node is initialized for servo + movement control."""
    global _node
    if _node is None:
        print("[DEBUG] Initializing robot_control_node...")
        if not rclpy.ok():
            print("[DEBUG] rclpy NOT initialized → calling rclpy.init()...")
            rclpy.init()
        _node = Node("robot_control_node")

        # Movement publisher
        _node.move_pub = _node.create_publisher(String, "movement_cmd", 10)
        print("[DEBUG] Publisher initialized → topic: movement_cmd")

        # Servo publisher
        _node.servo_pub = _node.create_publisher(UInt16, "servo_angle", 10)
        print("[DEBUG] Publisher initialized → topic: servo_angle")

    else:
        print("[DEBUG] robot_control_node already running")

    return _node

# ============================================================
#  SERVO TOOL (single servo, 0–180 degrees)
# ============================================================
@tool
def servo_tool(angle: int) -> str:
    """
    Move a single servo connected to ESP32 via micro-ROS.
    angle: 0–180 degrees
    """

    print(f"[DEBUG] servo_tool called → angle={angle}")

    if not (0 <= angle <= 180):
        print(f"[ERROR] Invalid angle: {angle}")
        return "Error: servo angle must be between 0 and 180"

    node = _ensure_node()

    msg = UInt16()
    msg.data = int(angle)

    print(f"[DEBUG] Publishing to /servo_angle → {angle}")
    node.servo_pub.publish(msg)

    final_msg = f"✅ Servo moved to {angle}°"
    print(f"[DEBUG] servo_tool returning: {final_msg}")
    return final_msg


# ============================================================
#  MOVEMENT TOOL ("F","B","L","R","S") with distance/degrees
# ============================================================
@tool
def move_robo(direction: str, value: Optional[float] = None) -> str:
    """
    Move robot using direction + value.

    direction:
        "F" → forward (value = meters or cm )
        "B" → backward (value = meters or cm)
        "L" → rotate left (value = degrees 0–360)
        "R" → rotate right (value = degrees 0–360)
        "S" → stop (no value needed)

    value: numeric argument for distance or degrees
    NOTE: if user provides it in cms please convert to meters internally.
    """

    print(f"[DEBUG] move_robo called → direction={direction}, value={value}")

    direction = direction.upper()
    valid = ["F", "B", "L", "R", "S"]

    if direction not in valid:
        print(f"[ERROR] Invalid direction: {direction}")
        return f"Error: direction must be one of {valid}"

    # Load node
    node = _ensure_node()

    # Speed constants — tune if needed
    forward_speed_cm_s = 28.0
    turn_speed_deg_s = 190.0
    duration = 0.0

    # -------------------
    # Compute Timer
    # -------------------
    if direction in ["F", "B"]:
        if value is None:
            return "Error: forward/backward needs value (meters or cm)."

        # Convert meters → cm if small
        distance_cm = value * 100 if value < 20 else value
        duration = distance_cm / forward_speed_cm_s
        print(f"[DEBUG] Computed duration for {distance_cm} cm → {duration:.2f}s")

    elif direction in ["L", "R"]:
        if value is None:
            return "Error: turning needs degrees."

        angle_deg = value
        duration = angle_deg / turn_speed_deg_s
        print(f"[DEBUG] Computed duration for {angle_deg}° → {duration:.2f}s")

    elif direction == "S":
        duration = 0

    # -------------------
    # Send movement command
    # -------------------
    msg = String()
    msg.data = direction

    print(f"[DEBUG] Publishing movement command → {direction}")
    node.move_pub.publish(msg)

    # Sleep for motion
    import time
    if duration > 0:
        print(f"[DEBUG] Sleeping for {duration:.2f}s to complete motion...")
        time.sleep(duration)

        # STOP after finishing
        stop_msg = String()
        stop_msg.data = "S"
        print("[DEBUG] Publishing → STOP")
        node.move_pub.publish(stop_msg)

    final_msg = f"🚗 Executed {direction} for {duration:.2f} seconds"
    print(f"[DEBUG] move_robo returning: {final_msg}")
    return final_msg

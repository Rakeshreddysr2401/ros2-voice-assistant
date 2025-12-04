@tool
def move_robo(direction: str, value: float = 0.0) -> str:
    """
    Move robot base.
    direction: 'F','B','L','R','S'
    value:
        - L,R: degrees
        - F,B: distance (cm or m)
        - S: ignored
    """

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import time

    direction = direction.upper()
    print(f"[TOOL] move_robo → {direction}, value={value}")

    valid = ["F","B","L","R","S"]
    if direction not in valid:
        return "Error: direction must be F,B,L,R,S."

    # Movement speeds (to tune)
    forward_speed_cm_s = 12.0
    turn_speed_deg_s = 180.0

    # Compute duration
    if direction in ["F","B"]:
        distance_cm = value * 100 if value < 10 else value
        duration = distance_cm / forward_speed_cm_s
    elif direction in ["L","R"]:
        duration = value / turn_speed_deg_s
    else:
        duration = 0

    # ROS init
    if not rclpy.ok():
        rclpy.init()

    node = Node("move_robo_tool")
    pub = node.create_publisher(String, "movement_cmd", 10)

    msg = String()
    msg.data = direction
    print(f"[TOOL] Publishing → {direction}")
    pub.publish(msg)

    if duration > 0:
        time.sleep(duration)

    msg.data = "S"
    print("[TOOL] Publishing → STOP")
    pub.publish(msg)

    return f"Executed {direction} for {duration:.2f} seconds."

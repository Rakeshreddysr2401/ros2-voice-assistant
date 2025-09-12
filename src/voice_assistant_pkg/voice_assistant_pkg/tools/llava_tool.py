import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from langchain_core.tools import tool

# Persistent ROS2 client node (fast)
rclpy.init(args=None)
node = Node("llava_tool_client")
client = node.create_client(Trigger, "llava_describe")

# Wait until service available
if not client.wait_for_service(timeout_sec=10.0):
    node.get_logger().error("⚠️ llava_describe service not available")
else:
    node.get_logger().info("✅ Connected to llava_describe service")

@tool
def describe_scene() -> str:
    """Use this to describe what the robot sees in front of it using LLaVA."""
    req = Trigger.Request()
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is not None and future.result().success:
        return future.result().message
    else:
        return f"Error: {future.result().message if future.result() else 'No response'}"

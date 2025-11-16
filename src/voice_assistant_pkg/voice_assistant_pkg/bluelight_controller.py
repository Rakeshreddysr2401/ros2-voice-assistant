import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import threading
import time


class BlueLightController(Node):
    def __init__(self):
        super().__init__('bluelight_controller')

        # Publishers
        self.cmd_pub = self.create_publisher(String, '/led_cmd', 10)
        self.hb_pub = self.create_publisher(Bool, '/heartbeat', 10)

        # Start heartbeat background thread
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        self.get_logger().info("BlueLightController node started.")

    def _heartbeat_loop(self):
        msg = Bool()
        msg.data = True
        while rclpy.ok():
            self.hb_pub.publish(msg)
            time.sleep(0.5)  # 2 Hz

    # OPTIONAL: callable from other modules
    def turn_on(self):
        msg = String()
        msg.data = "on"
        self.cmd_pub.publish(msg)
        self.get_logger().info("Blue LED → ON")

    def turn_off(self):
        msg = String()
        msg.data = "off"
        self.cmd_pub.publish(msg)
        self.get_logger().info("Blue LED → OFF")


def main(args=None):
    rclpy.init(args=args)
    node = BlueLightController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

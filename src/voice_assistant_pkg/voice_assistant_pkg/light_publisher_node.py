import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class TrafficPublisher(Node):
    def __init__(self):
        super().__init__('traffic_light_publisher')

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # CHANGE THIS LINE - add the /rt/ prefix:
        self.pub = self.create_publisher(String, '/rt/traffic_light', qos)

        self.get_logger().info("🚦 Traffic Light Publisher Node Started")
        self.get_logger().info("Publishing to topic: /rt/traffic_light")

        self.colors = ["red", "green", "orange"]
        self.index = 0

        self.timer = self.create_timer(5.0, self.timer_callback)

    def timer_callback(self):
        color = self.colors[self.index]
        self.index = (self.index + 1) % len(self.colors)
        msg = String()
        msg.data = color
        self.pub.publish(msg)
        self.get_logger().info(f"Sent: {color}")


def main(args=None):
    rclpy.init(args=args)
    node = TrafficPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TrafficPublisher(Node):
    def __init__(self):
        super().__init__('traffic_light_publisher')
        self.pub = self.create_publisher(String, 'traffic_light', 10)


    def send(self, color):
        msg = String()
        msg.data = color.upper()
        self.pub.publish(msg)
        self.get_logger().info(f"Sent traffic command: {color}")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

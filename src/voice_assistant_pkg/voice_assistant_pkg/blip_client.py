#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_interfaces.msg import VisualQuery
import threading
import time


class BlipClient(Node):
    def __init__(self):
        super().__init__("blip_client")

        # Publisher for queries and subscriber for responses
        self.query_pub = self.create_publisher(VisualQuery, "visual_query_request", 10)
        self.response_sub = self.create_subscription(VisualQuery, "visual_query_response", self.handle_response, 10)

        # Response storage
        self.latest_response = None
        self.response_received = threading.Event()

        self.get_logger().info("✅ BLIP Client ready!")

    def handle_response(self, msg):
        self.latest_response = msg
        self.response_received.set()

    def send_query(self, query=""):
        # Reset event
        self.response_received.clear()
        self.latest_response = None

        # Send query
        request = VisualQuery()
        request.query = query
        request.response = ""  # Empty for request

        self.query_pub.publish(request)

        if query.strip():
            print(f"🔍 Query sent: '{query}'")
        else:
            print("📷 Requesting image description...")

        # Wait for response
        if self.response_received.wait(timeout=10.0):
            if self.latest_response:
                print(f"💬 Response: {self.latest_response.response}\n")
                return self.latest_response.response
        else:
            print("⏰ Timeout waiting for response\n")
            return None


def main(args=None):
    rclpy.init(args=args)
    client = BlipClient()

    print("🤖 BLIP Visual Query Client")
    print("Commands:")
    print("  - Press Enter (empty) for image description")
    print("  - Type a query for visual question answering")
    print("  - Type 'quit' to exit\n")

    # Start spinning in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(client,), daemon=True)
    spin_thread.start()

    try:
        while True:
            query = input("Enter query (or press Enter for description): ").strip()

            if query.lower() == 'quit':
                break

            client.send_query(query)

    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
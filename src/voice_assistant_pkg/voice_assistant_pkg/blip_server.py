#!/usr/bin/env python3
import os
import cv2
import threading
import time
from PIL import Image
import rclpy
from rclpy.node import Node
from transformers import BlipProcessor, BlipForConditionalGeneration
from custom_interfaces.msg import VisualQuery


class BlipServer(Node):
    def __init__(self):
        super().__init__("blip_server")

        # Camera setup
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            exit(1)

        # Load BLIP model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Frame buffer
        self.latest_frame = None
        self.lock = threading.Lock()

        # Start camera thread
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Create subscriber for queries and publisher for responses
        self.query_sub = self.create_subscription(VisualQuery, "visual_query_request", self.handle_query, 10)
        self.response_pub = self.create_publisher(VisualQuery, "visual_query_response", 10)
        self.get_logger().info("🤖 BLIP Server ready (topics: visual_query_request/response)")

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def handle_query(self, request):
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        response_msg = VisualQuery()
        response_msg.query = request.query

        if frame is None:
            response_msg.response = "⚠️ No camera frame available"
            self.response_pub.publish(response_msg)
            return

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process based on query
        if not request.query.strip():  # Empty query - image description
            inputs = self.processor(images=image, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            result = self.processor.decode(out[0], skip_special_tokens=True)
            self.get_logger().info(f"Generated description: {result}")
        else:  # Non-empty query - conditional generation
            inputs = self.processor(images=image, text=request.query, return_tensors="pt")
            out = self.model.generate(**inputs, max_length=50)
            result = self.processor.decode(out[0], skip_special_tokens=True)
            self.get_logger().info(f"Query: '{request.query}' -> Response: {result}")

        response_msg.response = result
        self.response_pub.publish(response_msg)

    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlipServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import cv2
import threading
import time
from PIL import Image
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipServer(Node):
    def __init__(self):
        super().__init__("blip_server")

        # Camera setup
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.get_logger().info(f"Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            raise SystemExit(1)

        # Load BLIP captioning model
        self.get_logger().info("Loading BLIP captioning model...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.get_logger().info("✅ Captioning model loaded!")

        # Frame buffer and description cache
        self.latest_frame = None
        self.latest_description = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Subscribe to user_input to trigger description generation
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)

        # Trigger service
        self.describe_srv = self.create_service(Trigger, "blip_describe", self.handle_describe_service)

        self.get_logger().info("🤖 BLIP Captioning Server ready (service: /blip_describe, subscriber: /user_input)")

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def on_user_input(self, msg: String):
        """Generate image description when user input is received"""
        self.get_logger().info(f"User input received: {msg.data[:50]}... - Generating image description")
        self.generate_description()

    def generate_description(self):
        """Generate description from current frame"""
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            self.get_logger().warning("⚠️ No camera frame available for description")
            with self.lock:
                self.latest_description = None
            return

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.caption_processor(images=image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=30)
            result = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()

            with self.lock:
                self.latest_description = result

            self.get_logger().info(f"✅ Description ready: {result}")
        except Exception as e:
            self.get_logger().error(f"Error generating description: {e}")
            with self.lock:
                self.latest_description = None

    def handle_describe_service(self, request, response):
        """Service handler - returns pre-generated description"""
        with self.lock:
            description = self.latest_description

        if description is None:
            # Fallback: generate description on-demand if none exists
            self.get_logger().info("No pre-generated description, generating on-demand...")
            self.generate_description()
            with self.lock:
                description = self.latest_description

        if description is None:
            response.success = False
            response.message = "⚠️ No camera frame available or failed to generate description"
        else:
            response.success = True
            response.message = description
            self.get_logger().info(f"Returning cached description: {description}")

        return response

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
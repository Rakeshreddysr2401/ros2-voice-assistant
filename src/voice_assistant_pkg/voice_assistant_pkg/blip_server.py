#!/usr/bin/env python3
import os
import cv2
import threading
import numpy as np
import time
from PIL import Image
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipServer(Node):
    def __init__(self):
        super().__init__("blip_server")

        # Load BLIP captioning model
        self.get_logger().info("Loading BLIP captioning model...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.get_logger().info("✅ Captioning model loaded!")

        # Frame and description buffers
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_description = None
        self.lock = threading.Lock()

        # Subscribe to shared camera topic
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # Subscribe to user_input topic
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)

        # Create service
        self.describe_srv = self.create_service(Trigger, "blip_describe", self.handle_describe_service)

        self.get_logger().info("🤖 BLIP Captioning Server ready (subscribing to /camera/image_raw/compressed)")

    def image_callback(self, msg: CompressedImage):
        """Receive frames from shared camera"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding frame: {e}")

    def on_user_input(self, msg: String):
        """Generate caption when user speaks"""
        self.get_logger().info(f"User input received: {msg.data[:50]}... - Generating caption")
        self.generate_description()

    def generate_description(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            self.get_logger().warning("⚠️ No frame available for description")
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

            self.get_logger().info(f"🖼️ Description ready: {result}")

        except Exception as e:
            self.get_logger().error(f"Error generating description: {e}")
            with self.lock:
                self.latest_description = None

    def handle_describe_service(self, request, response):
        """Service handler - returns pre-generated description or generates on-demand"""
        with self.lock:
            desc = self.latest_description

        if desc is None:
            # Fallback: generate description on-demand if none exists
            self.get_logger().info("No pre-generated description, generating on-demand...")
            self.generate_description()
            with self.lock:
                desc = self.latest_description

        if desc:
            response.success = True
            response.message = desc
            self.get_logger().info(f"Returning description: {desc}")
        else:
            response.success = False
            response.message = "⚠️ No camera frame available or failed to generate description"

        return response


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
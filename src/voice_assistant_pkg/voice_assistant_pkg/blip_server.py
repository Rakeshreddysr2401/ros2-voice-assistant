#!/usr/bin/env python3
import os
import cv2
import threading
import time
from PIL import Image
import rclpy
from rclpy.node import Node
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

        # Frame buffer
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Trigger service
        self.describe_srv = self.create_service(Trigger, "blip_describe", self.handle_describe_service)

        self.get_logger().info("🤖 BLIP Captioning Server ready (service: /blip_describe)")

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def handle_describe_service(self, request, response):
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            response.success = False
            response.message = "⚠️ No camera frame available"
            return response

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.caption_processor(images=image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=30)
            result = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
            response.success = True
            response.message = result
            self.get_logger().info(f"Caption generated: {result}")
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            self.get_logger().error(f"Processing error: {e}")

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

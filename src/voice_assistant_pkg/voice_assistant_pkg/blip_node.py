#!/usr/bin/env python3
import os
import cv2
import threading
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import time


class BlipNode(Node):
    def __init__(self):
        super().__init__("blip_node")

        # Camera source
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            exit(1)
        self.get_logger().info("✅ Camera opened successfully for BLIP.")

        # Load BLIP model + processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Frame buffer
        self.latest_frame = None
        self.last_timestamp = 0
        self.lock = threading.Lock()

        # Start camera thread
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Service
        self.srv = self.create_service(Trigger, "blip_describe", self.handle_describe)

    def camera_loop(self):
        """Continuously grab frames in the background"""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
                    self.last_timestamp = time.time()

    def handle_describe(self, request, response):
        """Handle service call and ensure we get a *fresh* frame"""
        start_time = time.time()
        current_frame = None
        current_ts = 0

        # Wait until we get a frame newer than last processed
        while time.time() - start_time < 2.0:  # max wait 2s
            with self.lock:
                if self.latest_frame is not None and self.last_timestamp > current_ts:
                    current_frame = self.latest_frame.copy()
                    current_ts = self.last_timestamp
                    break
            time.sleep(0.05)

        if current_frame is None:
            response.success = False
            response.message = "⚠️ No fresh frame available"
            self.get_logger().warn(response.message)
            return response

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(image, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_length=50)
        description = self.processor.decode(out[0], skip_special_tokens=True)

        response.success = True
        response.message = description
        self.get_logger().info(f"📝 Scene: {description}")
        return response

    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

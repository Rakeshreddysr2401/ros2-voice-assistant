#!/usr/bin/env python3
import os
import cv2
import threading
import time
from PIL import Image
import torch
from transformers import GitProcessor, GitForCausalLM
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class LightweightVisionServer(Node):
    def __init__(self):
        super().__init__("lightweight_vision_server")

        # ---------------- Camera setup ----------------
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)
        self.get_logger().info(f"Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            raise SystemExit(1)

        # ---------------- Model setup (GIT - lightweight) ----------------
        self.get_logger().info("Loading GIT model (lightweight)...")
        self.processor = GitProcessor.from_pretrained("microsoft/git-base")
        self.model = GitForCausalLM.from_pretrained("microsoft/git-base")
        self.model.to("cpu")  # Force CPU
        self.model.eval()
        self.get_logger().info("✅ GIT model loaded (139M params)")

        # ---------------- Frame buffer ----------------
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # ---------------- ROS2 Service ----------------
        self.describe_srv = self.create_service(
            Trigger, "vision_describe", self.handle_describe_service
        )
        self.get_logger().info("🔍 Lightweight Vision Server ready (service: /vision_describe)")

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
            # Convert to PIL
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ---------------- GIT inference ----------------
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to("cpu")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=4,
                    num_return_sequences=1
                )

            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
    node = LightweightVisionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

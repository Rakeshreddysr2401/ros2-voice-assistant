#!/usr/bin/env python3
import os
import io
import cv2
import json
import threading
import requests
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from custom_interfaces.srv import QwenVision

class MoondreamServer(Node):
    def __init__(self):
        super().__init__("moondream_server")

        # Ollama configuration
        self.ollama_host = os.getenv("OLLAMA_HOST", "192.168.1.22")
        self.ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        self.model = os.getenv("MOONDREAM_MODEL", "moondream") # Specifically for Moondream

        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()

        # Subscribe to shared camera
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # ROS service
        self.srv = self.create_service(QwenVision, "moondream_vision_describe", self.handle_service)

        self.get_logger().info(f"🚀 Moondream FAST Vision Server ready (model: {self.model})")

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding frame: {e}")

    def handle_service(self, request, response):
        query = request.query
        self.get_logger().info(f"Fast Vision Query: {query}")

        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            response.response = "No frame available."
            return response

        # Convert to PIL for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Call Ollama
        result = self.call_ollama(query, image)
        response.response = result if result else "Failed to get response from Moondream."
        return response

    def call_ollama(self, prompt: str, pil_img: Image.Image):
        # Convert PIL to Base64
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        img_b64 = b64_image = io.BytesIO(buff.getvalue()).read()
        import base64
        img_b64 = base64.b64encode(img_b64).decode('utf-8')

        url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "images": [img_b64]
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            self.get_logger().error(f"Ollama error: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = MoondreamServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

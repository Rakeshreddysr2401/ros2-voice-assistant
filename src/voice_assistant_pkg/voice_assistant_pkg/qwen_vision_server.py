import os
import io
import cv2
import time
import base64
import json
import threading
import requests
from PIL import Image

import rclpy
from rclpy.node import Node
from custom_interfaces.srv import QwenVision   # <-- NEW import


class QwenVisionServer(Node):
    def __init__(self):
        super().__init__("qwen_vision_server")

        # Ollama configuration
        self.ollama_host = os.getenv("OLLAMA_HOST", "192.168.1.22")
        self.ollama_port = int(os.getenv("OLLAMA_PORT", 11434))
        self.model = os.getenv("MODEL", "qwen2.5vl:3b")

        # Camera setup
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.get_logger().info(f"Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera {camera_source}")
            raise SystemExit(1)

        # Frame buffers
        self.latest_frame = None
        self.first_frame = None
        self.lock = threading.Lock()
        self.running = True

        # Start camera thread
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # ROS service
        self.qwen_srv = self.create_service(QwenVision, "qwen_vision_describe", self.handle_qwen_service)

        self.get_logger().info("🤖 Qwen Vision Server ready (service: /qwen_vision_describe)")

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
                    if self.first_frame is None:
                        self.first_frame = frame.copy()
            time.sleep(0.05)

    def pil_image_to_b64(self, pil_img: Image.Image) -> str:
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG", quality=90)
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    def call_ollama_generate(self, prompt: str, image_b64: str | None = None):
        url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
        payload = {"model": self.model, "prompt": prompt}
        if image_b64:
            if image_b64.startswith("data:image"):
                image_b64 = image_b64.split(",")[1]
            payload["images"] = [image_b64]

        try:
            resp_text = ""
            resp = requests.post(url, json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if data.get("response"):
                        resp_text += data["response"]
                except json.JSONDecodeError:
                    decoded = line.decode("utf-8")
                    if '"response":"' in decoded:
                        resp_text += decoded.split('"response":"')[1].split('"')[0]
            return resp_text.strip() if resp_text else None
        except Exception as e:
            self.get_logger().error(f"Error calling Ollama: {e}")
            return None

    def generate_description(self, query: str, frame):
        if frame is None:
            self.get_logger().warning("⚠️ No frame available for description")
            return None

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_b64 = self.pil_image_to_b64(image)
            prompt = query if query else "Describe the following image."

            self.get_logger().info("Sending image to Ollama...")
            resp = self.call_ollama_generate(prompt, image_b64=img_b64)
            return resp.strip() if resp else None
        except Exception as e:
            self.get_logger().error(f"Error generating description: {e}")
            return None

    def handle_qwen_service(self, request, response):
        """Service callback for QwenVision.srv"""
        with self.lock:
            frame = self.latest_frame.copy() if request.use_latest else (
                self.first_frame.copy() if self.first_frame is not None else None
            )

        description = self.generate_description(request.query, frame)

        if description is None:
            response.success = False
            response.description = "⚠️ Failed to generate description"
        else:
            response.success = True
            response.description = description
            self.get_logger().info(f"✅ Returning description: {description[:200]}")

        return response

    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QwenVisionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

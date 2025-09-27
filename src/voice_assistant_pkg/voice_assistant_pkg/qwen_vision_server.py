# qwen_vision_server.py
"""
ROS2 node that captures camera frames, primes Ollama's qwen2.5vl:7b with the image (so Ollama caches image tokens),
returns a cached description via a Trigger service, and answers follow-up text queries via a ROS topic.

Services / topics:
 - Service: /qwen_vision_describe  (std_srvs.srv.Trigger) -> returns cached description (primes if missing)
 - Topic (sub): /vision_query (std_msgs.msg.String) -> server will answer follow-up queries that assume image already sent to Ollama
 - Topic (pub): /vision_response (std_msgs.msg.String) -> server publishes answer to the corresponding query

Environment variables:
 - OLLAMA_HOST (default: "localhost")
 - OLLAMA_PORT (default: 11434)
 - MODEL (default: "qwen2.5vl:7b")
 - CAMERA_SOURCE (default: "0")

NOTE: This implements a pragmatic flow: image is sent once to Ollama during description generation. When follow-up queries arrive they are sent as text only (no image) so Ollama uses its cached image tokens (if available) and responds faster.
"""

import os
import io
import time
import threading
import base64
import requests
import json
from PIL import Image
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class QwenVisionServer(Node):
    def __init__(self):
        super().__init__('qwen_vision_server')

        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = int(os.getenv('OLLAMA_PORT', 11434))
        self.model = os.getenv('MODEL', 'qwen2.5vl:7b')
        camera_source = os.getenv('CAMERA_SOURCE', '0')
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.get_logger().info(f"Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera {camera_source}")
            raise SystemExit(1)

        self.latest_frame = None
        self.latest_description = None
        self.primed = False  # True after image has been sent to ollama
        self.lock = threading.Lock()
        self.running = True

        # Start camera thread
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # ROS interfaces
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)
        self.query_sub = self.create_subscription(String, 'vision_query', self.on_vision_query, 10)
        self.response_pub = self.create_publisher(String, 'vision_response', 10)
        self.describe_srv = self.create_service(Trigger, 'qwen_vision_describe', self.handle_describe_service)

        self.get_logger().info('Qwen Vision Server ready (service: /qwen_vision_describe, topics: /vision_query -> /vision_response)')

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def on_user_input(self, msg: String):
        self.get_logger().info('User input received -> generating image description (prime Ollama)')
        self.generate_description()

    def pil_image_to_b64(self, pil_img: Image.Image) -> str:
        buff = io.BytesIO()
        pil_img.save(buff, format='JPEG', quality=90)
        b = buff.getvalue()
        return base64.b64encode(b).decode('utf-8')

    def call_ollama_generate(self, prompt: str, images_b64: list | None = None, stream: bool = False,
                             timeout: int = 30):
        url = f'http://{self.ollama_host}:{self.ollama_port}/api/generate'
        payload = {'model': self.model, 'prompt': prompt}
        if images_b64:
            payload['images'] = images_b64

        try:
            if stream:
                # Explicit streaming generator
                def stream_generator():
                    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if data.get('response'):
                                    yield data['response']
                            except Exception:
                                continue
                return stream_generator()
            else:
                # Non-streaming: collect into string
                with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                    resp.raise_for_status()
                    output = []
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if data.get('response'):
                                output.append(data['response'])
                        except Exception:
                            continue
                    return "".join(output).strip() if output else None
        except Exception as e:
            self.get_logger().error(f'Error calling Ollama: {e}')
            return None

    def generate_description(self):
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            self.get_logger().warning('No camera frame available for description')
            with self.lock:
                self.latest_description = None
            return

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_b64 = self.pil_image_to_b64(image)

            system_prompt = (
                "You are an assistant that describes images. Produce a concise natural-language description followed by a JSON block labelled METADATA with objects, approximate colors, and bounding descriptions. "
                "Keep the natural description as the primary answer. The JSON METADATA should be machine-parseable."
            )

            full_prompt = system_prompt + "\n\nDescribe the following image:"

            self.get_logger().info('Sending image to Ollama to prime model and get initial description...')
            resp = self.call_ollama_generate(full_prompt, images_b64=[img_b64], stream=False, timeout=60)

            if not resp:
                self.get_logger().error('Ollama did not return a description')
                with self.lock:
                    self.latest_description = None
                return

            with self.lock:
                self.latest_description = resp.strip()
                self.primed = True

            self.get_logger().info(f'Description ready and model primed: {self.latest_description[:200]}')
        except Exception as e:
            self.get_logger().error(f'Error generating description: {e}')
            with self.lock:
                self.latest_description = None
                self.primed = False

    def handle_describe_service(self, request, response):
        with self.lock:
            description = self.latest_description

        if description is None:
            self.get_logger().info('No cached description available — generating on-demand')
            self.generate_description()
            with self.lock:
                description = self.latest_description

        if description is None:
            response.success = False
            response.message = 'No camera frame available or failed to generate description'
        else:
            response.success = True
            response.message = description
            self.get_logger().info('Returning cached description')

        return response

    def on_vision_query(self, msg: String):
        query = msg.data.strip()
        self.get_logger().info(f'Received vision query: {query[:120]}')

        if not self.primed:
            self.get_logger().info('Model not primed with an image yet — priming now (this may take a few seconds)')
            self.generate_description()

        followup_prompt = (
            "You have previously been shown a specific image (already in your context). "
            "Answer the user's question about that image concisely and include any short JSON if requested.\n\nUser question: " + query
        )

        resp = self.call_ollama_generate(followup_prompt, images_b64=None, stream=False, timeout=30)
        if not resp:
            resp_text = 'Failed to get reply from vision model.'
        else:
            resp_text = resp.strip()

        out_msg = String()
        out_msg.data = resp_text
        self.response_pub.publish(out_msg)
        self.get_logger().info('Published vision_response')

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()

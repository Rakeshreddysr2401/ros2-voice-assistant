#!/usr/bin/env python3
import os
import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from custom_interfaces.srv import YoloDetect  # your custom service

class YoloServer(Node):
    def __init__(self):
        super().__init__("yolo_server")

        # Load YOLO model
        self.get_logger().info("Loading YOLOv8s model...")
        self.yolo_model = YOLO("yolov8s.pt")
        self.get_logger().info("✅ YOLO model loaded!")

        # Frame buffers
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_detections = []
        self.lock = threading.Lock()

        # Subscribe to shared camera topic
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # Subscribe to user input
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)

        # Custom service
        self.detect_srv = self.create_service(YoloDetect, "yolo_detect", self.handle_detect_service)

        self.get_logger().info("🤖 YOLO Detection Server ready (listening on /camera/image_raw/compressed)")

    def image_callback(self, msg: CompressedImage):
        """Receive shared camera frames"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with self.lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding image: {e}")

    def on_user_input(self, msg: String):
        self.get_logger().info(f"User input received: {msg.data[:50]}... - Running YOLO detection")
        self.run_yolo_detection()

    def run_yolo_detection(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            self.get_logger().warning("⚠️ No frame available for YOLO detection")
            with self.lock:
                self.latest_detections = []
            return

        try:
            results = self.yolo_model.predict(source=frame, conf=0.5, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = r.names[cls_id]
                    conf = float(box.conf)
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]
                    detections.append({"label": label, "confidence": conf, "bbox": xyxy})

            with self.lock:
                self.latest_detections = detections

            self.get_logger().info(f"✅ Detections ready: {detections}")

        except Exception as e:
            self.get_logger().error(f"Error running YOLO detection: {e}")
            with self.lock:
                self.latest_detections = []

    def handle_detect_service(self, request, response):
        """Custom service to run detection with on-demand fallback"""
        with self.lock:
            detections = self.latest_detections

        if not detections:
            # Fallback: generate detections on-demand if none exist
            self.get_logger().info("No pre-generated detections, running detection on-demand...")
            self.run_yolo_detection()
            with self.lock:
                detections = self.latest_detections

        if not detections:
            response.success = False
            response.message = "⚠️ No objects detected or no frame available"
        else:
            response.success = True
            response.message = str(detections)
            self.get_logger().info(f"Returning detections: {detections}")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = YoloServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
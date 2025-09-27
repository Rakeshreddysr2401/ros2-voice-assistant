import os
import cv2
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ultralytics import YOLO


class YoloServer(Node):
    def __init__(self):
        super().__init__("yolo_server")

        # Camera setup
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.get_logger().info(f"Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            raise SystemExit(1)

        # Load YOLO model
        self.get_logger().info("Loading YOLOv8s model...")
        self.yolo_model = YOLO("yolov8s.pt")  # You can change to yolov8s.pt for better accuracy
        self.get_logger().info("✅ YOLOv8n model loaded!")

        # Frame buffer and detection cache
        self.latest_frame = None
        self.latest_detections = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Subscribe to user_input to trigger detection
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)

        # Trigger service
        self.detect_srv = self.create_service(Trigger, "yolo_detect", self.handle_detect_service)

        self.get_logger().info("🤖 YOLO Detection Server ready (service: /yolo_detect, subscriber: /user_input)")

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def on_user_input(self, msg: String):
        """Run YOLO detection when user input is received"""
        self.get_logger().info(f"User input received: {msg.data[:50]}... - Running YOLO detection")
        self.run_yolo_detection()

    def run_yolo_detection(self):
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            self.get_logger().warning("⚠️ No camera frame available for YOLO detection")
            with self.lock:
                self.latest_detections = []
            return

        try:
            # Ensure frame is valid BGR
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.get_logger().warning("⚠️ Invalid frame shape, skipping detection")
                return

            # Run YOLO with lower confidence threshold
            results = self.yolo_model.predict(source=frame, conf=0.5, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = r.names[cls_id]
                    conf = float(box.conf)
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]  # [x1, y1, x2, y2]
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": xyxy
                    })

            with self.lock:
                self.latest_detections = detections

            self.get_logger().info(f"✅ Detections ready: {detections}")

        except Exception as e:
            self.get_logger().error(f"Error running YOLO detection: {e}")
            with self.lock:
                self.latest_detections = []

    def handle_detect_service(self, request, response):
        """Service handler - returns pre-generated detections"""
        with self.lock:
            detections = self.latest_detections

        if detections is None or len(detections) == 0:
            # Fallback: run detection on-demand
            self.get_logger().info("No pre-generated detections, running on-demand...")
            self.run_yolo_detection()
            with self.lock:
                detections = self.latest_detections

        if detections is None or len(detections) == 0:
            response.success = False
            response.message = "⚠️ No objects detected"
        else:
            response.success = True
            response.message = str(detections)
            self.get_logger().info(f"Returning cached detections: {detections}")

        return response

    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


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

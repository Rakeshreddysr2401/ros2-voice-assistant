#!/usr/bin/env python3
import os
import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # ROS2 publisher
        self.pub = self.create_publisher(String, "camera_detections", 10)

        # Camera source (0 = default webcam; or USB/IP cam from env)
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        # Init camera
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            exit(1)

        self.get_logger().info("✅ Camera opened successfully (headless mode).")

        # Load small YOLO model (lightweight)
        self.model = YOLO("yolov8n.pt")

        # Timer: run detection every 0.2s
        self.timer = self.create_timer(0.2, self.detect_objects)

    def detect_objects(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("⚠️ Failed to grab frame")
            return

        results = self.model(frame, verbose=False)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

        msg = String()
        msg.data = json.dumps({"detections": detections})
        self.pub.publish(msg)

        if detections:
            self.get_logger().info(f"📦 {detections}")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

import os
import cv2
import threading
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_interfaces.srv import YoloDetect  # <-- custom service

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
        self.yolo_model = YOLO("yolov8s.pt")
        self.get_logger().info("✅ YOLO model loaded!")

        # Frame buffers
        self.latest_frame = None
        self.first_frame = None
        self.latest_detections = []
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Subscribe to user input
        self.input_sub = self.create_subscription(String, 'user_input', self.on_user_input, 10)

        # Custom service
        self.detect_srv = self.create_service(YoloDetect, "yolo_detect", self.handle_detect_service)

        self.get_logger().info("🤖 YOLO Detection Server ready (service: /yolo_detect, subscriber: /user_input)")

    # ---------------- Camera Loop ----------------
    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
                    if self.first_frame is None:
                        self.first_frame = frame.copy()
            time.sleep(0.05)

    # ---------------- User Input ----------------
    def on_user_input(self, msg: String):
        self.get_logger().info(f"User input received: {msg.data[:50]}... - Running YOLO detection")
        self.run_yolo_detection(use_latest=True)

    # ---------------- Detection ----------------
    def run_yolo_detection(self, use_latest=True):
        with self.lock:
            if use_latest:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if frame is not None:
                    self.first_frame = frame.copy()
            else:
                frame = self.first_frame.copy() if self.first_frame is not None else None

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

    # ---------------- Service ----------------
    def handle_detect_service(self, request, response):
        use_latest = request.use_latest
        self.get_logger().info(f"YOLO detect request received: use_latest={use_latest}")

        self.run_yolo_detection(use_latest=use_latest)
        with self.lock:
            detections = self.latest_detections

        if not detections:
            response.success = False
            response.message = "⚠️ No objects detected"
        else:
            response.success = True
            response.message = str(detections)

        return response

    # ---------------- Cleanup ----------------
    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

# ---------------- Main ----------------
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

#!/usr/bin/env python3
import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import time


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        # Get camera source (default /dev/video0)
        camera_source = os.getenv("CAMERA_SOURCEZ", "/dev/video0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.get_logger().info(f"📷 Connecting to camera: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            raise SystemExit(1)

        # Set resolution (optional — adjust as needed)
        width = int(os.getenv("CAMERA_WIDTH", "640"))
        height = int(os.getenv("CAMERA_HEIGHT", "480"))
        fps = int(os.getenv("CAMERA_FPS", "10"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)

        self.timer_period = 1.0 / fps
        self.timer = self.create_timer(self.timer_period, self.publish_frame)
        self.get_logger().info(f"✅ Camera publisher started at {fps} FPS ({width}x{height})")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("⚠️ Failed to grab frame")
            return

        try:
            msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

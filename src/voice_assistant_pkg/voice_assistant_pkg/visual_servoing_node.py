#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from custom_interfaces.srv import YoloDetect
import json
import threading
import time

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')
        
        # Parameters
        self.target_label = ""
        self.is_active = False
        self.is_searching = False
        self.search_start_time = 0
        self.search_timeout = 15.0 # Search for 15s (about one full rotation)
        
        self.frame_width = 640
        self.center_x = self.frame_width / 2
        self.deadzone = 40  # pixels
        self.target_area_ratio = 0.25
        
        # Publishers
        self.cmd_pub = self.create_publisher(String, 'movement_cmd', 10)
        self.status_pub = self.create_publisher(String, 'agent_status', 10)
        
        # Subscriptions
        self.sub_control = self.create_subscription(String, 'servoing_control', self.control_callback, 10)
        
        # Service client for YOLO
        self.yolo_client = self.create_client(YoloDetect, 'yolo_detect')
        
        # Timer for the control loop (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("🎯 Visual Servoing Node Ready with Autonomous Search.")

    def control_callback(self, msg):
        # Format: "START:label" or "STOP"
        data = msg.data.split(':')
        command = data[0]
        
        if command == "START" and len(data) > 1:
            self.target_label = data[1]
            self.is_active = True
            self.is_searching = False
            self.get_logger().info(f"Tracking started for: {self.target_label}")
        else:
            self.is_active = False
            self.is_searching = False
            self.stop_robot()
            self.get_logger().info("Tracking stopped.")

    def stop_robot(self):
        msg = String()
        msg.data = "S"
        self.cmd_pub.publish(msg)

    def control_loop(self):
        if not self.is_active:
            return

        # If we are searching, we send rotation commands periodically
        if self.is_searching:
            if time.time() - self.search_start_time > self.search_timeout:
                self.get_logger().info("Search timeout reached.")
                self.is_active = False
                self.is_searching = False
                self.stop_robot()
                self.publish_status(f"Search failed. {self.target_label} not found.")
                return
            
            # Send a slow rotate command to scan
            msg = String()
            msg.data = "L" 
            self.cmd_pub.publish(msg)
            self.publish_status(f"Scanning for {self.target_label}...")

        if not self.yolo_client.service_is_ready():
            return

        # Request latest detections
        req = YoloDetect.Request()
        req.use_latest = True
        
        future = self.yolo_client.call_async(req)
        future.add_done_callback(self.process_detections)

    def process_detections(self, future):
        try:
            response = future.result()
            if not response.success:
                return

            detections = eval(response.message)
            target = None
            for d in detections:
                if d['label'].lower() == self.target_label.lower():
                    target = d
                    break

            if not target:
                if not self.is_searching:
                    self.get_logger().info("Target lost. Entering Autonomous Search mode.")
                    self.is_searching = True
                    self.search_start_time = time.time()
                return

            # Target Found! (Resume normal tracking)
            if self.is_searching:
                self.get_logger().info(f"Target {self.target_label} rediscovered!")
                self.is_searching = False

            bbox = target['bbox']
            obj_center_x = (bbox[0] + bbox[2]) / 2
            area_ratio = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / (640 * 480)
            
            offset_x = obj_center_x - self.center_x
            
            msg = String()
            if abs(offset_x) > self.deadzone:
                msg.data = "R" if offset_x > 0 else "L"
                self.cmd_pub.publish(msg)
                self.publish_status(f"Centering {self.target_label}...")
            elif area_ratio < self.target_area_ratio:
                msg.data = "F"
                self.cmd_pub.publish(msg)
                self.publish_status(f"Approaching {self.target_label}...")
            else:
                self.is_active = False
                self.stop_robot()
                self.publish_status(f"Arrived at {self.target_label}!")
                
        except Exception as e:
            self.get_logger().error(f"Error in VS loop: {e}")

    def publish_status(self, text):
        msg = String()
        msg.data = f"[Servoing] {text}"
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

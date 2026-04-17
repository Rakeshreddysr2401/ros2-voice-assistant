#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import math
import time

class SemanticMapNode(Node):
    def __init__(self):
        super().__init__('semantic_map_node')
        
        # Robot State (Relative to start position)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 # Orientation in radians
        
        # Approximate calibration for dead reckoning
        self.forward_speed = 0.12 # m/s
        self.turn_speed = math.radians(180) # rad/s (one full rotation in 2 seconds)
        
        self.last_update = time.time()
        self.current_cmd = "S"
        
        # Semantic Memory: { "label": (x, y) }
        self.object_map = {}
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'agent_status', 10)
        
        # Subscriptions
        self.create_subscription(String, 'movement_cmd', self.move_cb, 10)
        self.create_subscription(String, 'pin_object', self.pin_cb, 10)
        self.create_subscription(String, 'query_map', self.query_cb, 10)
        
        # 20Hz update timer for odometry
        self.create_timer(0.05, self.update_odometry)
        
        self.get_logger().info("🗺️ Semantic Mapping Node Ready (Dead Reckoning).")

    def move_cb(self, msg):
        self.current_cmd = msg.data.upper()

    def pin_cb(self, msg):
        label = msg.data.lower()
        self.object_map[label] = (self.x, self.y)
        self.get_logger().info(f"📍 Pinned '{label}' at ({self.x:.2f}, {self.y:.2f})")
        self.publish_status(f"Saved {label} to map at coordinates ({self.x:.2f}, {self.y:.2f})")

    def query_cb(self, msg):
        if not self.object_map:
            self.publish_status("My map is empty.")
            return

        report = ["My current map:"]
        for label, (ox, oy) in self.object_map.items():
            # Calculate distance and bearing from current position
            dx = ox - self.x
            dy = oy - self.y
            dist = math.sqrt(dx**2 + dy**2)
            # Bearing relative to current heading
            global_angle = math.atan2(dy, dx)
            relative_angle = math.degrees(global_angle - self.theta)
            
            # Normalize angle to -180 to 180
            while relative_angle > 180: relative_angle -= 360
            while relative_angle < -180: relative_angle += 360
            
            direction = "ahead"
            if relative_angle > 30: direction = "to my left"
            if relative_angle < -30: direction = "to my right"
            if abs(relative_angle) > 150: direction = "behind me"
            
            report.append(f"- {label}: {dist:.2f} meters {direction} ({relative_angle:.0f} degrees)")
        
        self.publish_status("\n".join(report))

    def update_odometry(self):
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        if self.current_cmd == "F":
            self.x += self.forward_speed * dt * math.cos(self.theta)
            self.y += self.forward_speed * dt * math.sin(self.theta)
        elif self.current_cmd == "B":
            self.x -= self.forward_speed * dt * math.cos(self.theta)
            self.y -= self.forward_speed * dt * math.sin(self.theta)
        elif self.current_cmd == "L":
            self.theta += self.turn_speed * dt
        elif self.current_cmd == "R":
            self.theta -= self.turn_speed * dt
            
    def publish_status(self, text):
        msg = String()
        msg.data = f"[MAP] {text}"
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

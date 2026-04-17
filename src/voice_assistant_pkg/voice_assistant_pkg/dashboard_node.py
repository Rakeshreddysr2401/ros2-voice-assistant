#!/usr/bin/env python3
import os
import cv2
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# ---------------------------------------------------------
# GLOBAL DATA
# ---------------------------------------------------------
latest_frame = None
agent_status = "Waiting for command..."
chat_history = []
frame_lock = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; display: flex; height: 100vh; }
        #left-panel { width: 60%; padding: 20px; display: flex; flex-direction: column; border-right: 1px solid #333; }
        #right-panel { width: 40%; padding: 20px; display: flex; flex-direction: column; }
        .video-container { width: 100%; background: #000; border-radius: 10px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        .video-container img { width: 100%; display: block; }
        .status-box { margin-top: 20px; background: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 4px solid #00e5ff; }
        .status-label { font-size: 0.8rem; color: #888; text-transform: uppercase; margin-bottom: 5px; }
        .status-text { font-size: 1.2rem; font-weight: bold; color: #00e5ff; }
        .chat-container { flex-grow: 1; background: #1e1e1e; border-radius: 8px; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .message { padding: 10px 15px; border-radius: 15px; max-width: 80%; line-height: 1.4; }
        .user { align-self: flex-end; background: #3d5afe; color: white; border-bottom-right-radius: 2px; }
        .bot { align-self: flex-start; background: #333; color: #e0e0e0; border-bottom-left-radius: 2px; }
        h2 { margin-top: 0; color: #fff; font-size: 1.2rem; }
    </style>
    <script>
        function updateData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerText = data.status;
                    const chat = document.getElementById('chat');
                    chat.innerHTML = '';
                    data.history.forEach(msg => {
                        const div = document.createElement('div');
                        div.className = 'message ' + (msg.role === 'user' ? 'user' : 'bot');
                        div.innerText = msg.text;
                        chat.appendChild(div);
                    });
                    chat.scrollTop = chat.scrollHeight;
                });
        }
        setInterval(updateData, 1000);
    </script>
</head>
<body>
    <div id="left-panel">
        <h2>Live Visual Feed</h2>
        <div class="video-container">
            <img src="/stream.mjpg" />
        </div>
        <div class="status-box">
            <div class="status-label">Current Thought / Action</div>
            <div id="status" class="status-text">Connecting...</div>
        </div>
    </div>
    <div id="right-panel">
        <h2>Conversation History</h2>
        <div id="chat" class="chat-container"></div>
    </div>
</body>
</html>
"""

# ---------------------------------------------------------
# HTTP SERVER LOGIC
# ---------------------------------------------------------
class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame, agent_status, chat_history
        
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
            
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"status": agent_status, "history": chat_history[-10:]}
            self.wfile.write(json.dumps(data).encode())

        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if latest_frame is None:
                            time.sleep(0.1)
                            continue
                        ret, jpeg = cv2.imencode('.jpg', latest_frame)
                        content = jpeg.tobytes()

                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.1)
            except Exception as e:
                print(f"Stream disconnected: {e}")

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

# ---------------------------------------------------------
# ROS2 NODE
# ---------------------------------------------------------
class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        
        # Subscriptions
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.image_cb, 10)
        self.create_subscription(String, 'agent_status', self.status_cb, 10)
        self.create_subscription(String, 'user_input', self.user_cb, 10)
        self.create_subscription(String, 'agent_response', self.bot_cb, 10)

        self.get_logger().info("🖥️ Dashboard Node started on http://localhost:8080")

    def image_cb(self, msg):
        global latest_frame
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with frame_lock:
                latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Error decoding image: {e}")

    def status_cb(self, msg):
        global agent_status
        agent_status = msg.data

    def user_cb(self, msg):
        global chat_history
        chat_history.append({"role": "user", "text": msg.data})

    def bot_cb(self, msg):
        global chat_history
        chat_history.append({"role": "bot", "text": msg.data})

def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    
    # Run HTTP server in thread
    server = ThreadedHTTPServer(('0.0.0.0', 8080), CamHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
scene_service.py
ROS2 service: /analyze_scene (voice_assistant_pkg/srv/AnalyzeScene)
Request: description: bool, coordinates: bool
Response: success: bool, message: string (JSON)
- Lazy loads YOLO and BLIP.
- Supports CAMERA_SOURCE env var:
    - If startswith "http" -> treated as IP camera (cv2.VideoCapture(url))
    - Else -> device path like "/dev/video0" or numeric "0"
- Resize frames to CAMERA_WIDTHxCAMERA_HEIGHT (env vars; defaults 320x240)
"""
import os
import json
import time
import traceback

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from PIL import Image

# Replace with your generated srv import
from voice_assistant_pkg.srv import AnalyzeScene


class SceneService(Node):
    def __init__(self):
        super().__init__('scene_service')
        # Config from env
        self.camera_source = os.getenv('CAMERA_SOURCE', '0')  # '0' or '/dev/video0' or 'http://...'
        self.width = int(os.getenv('CAMERA_WIDTH', '320'))
        self.height = int(os.getenv('CAMERA_HEIGHT', '240'))
        self.max_objects = int(os.getenv('MAX_OBJECTS', '12'))
        self.conf_threshold = float(os.getenv('DETECT_CONF', '0.25'))

        # Models (lazy)
        self.yolo = None
        self.blip = None
        self.blip_processor = None

        # Setup capture (lazy open)
        self.cap = None
        self.open_capture()

        # Create service
        self.srv = self.create_service(AnalyzeScene, 'analyze_scene', self.analyze_cb)
        self.get_logger().info(f"scene_service ready on /analyze_scene (camera={self.camera_source})")

    # ---------------- Camera ----------------
    def open_capture(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            # If URL
            if str(self.camera_source).lower().startswith('http'):
                src = self.camera_source
            else:
                # numeric index or device path
                if str(self.camera_source).isdigit():
                    src = int(self.camera_source)
                else:
                    src = self.camera_source

            self.get_logger().info(f"Opening camera source: {src}")
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if isinstance(src, int) or str(src).startswith('/dev') else 0)
            # Try to set resolution (some drivers ignore)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            time.sleep(0.2)
            if not self.cap.isOpened():
                # fallback: try without backend hint
                self.get_logger().warn("Initial open failed, retrying without backend hint")
                self.cap = cv2.VideoCapture(src)
                time.sleep(0.2)
        except Exception as e:
            self.get_logger().error(f"open_capture error: {e}")
            self.cap = None

    def capture_frame(self, timeout=3.0):
        if self.cap is None or not self.cap.isOpened():
            self.open_capture()
            if self.cap is None or not self.cap.isOpened():
                return None
        # try to read several times
        t0 = time.time()
        while time.time() - t0 < timeout:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # convert BGR->RGB for PIL/BLIP, but we keep BGR for OpenCV/Yolo
                # Resize to configured size to save memory/time
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                return frame
            time.sleep(0.05)
        return None

    # ---------------- YOLO (detection) ----------------
    def load_yolo(self):
        if self.yolo is not None:
            return
        try:
            # Prefer ultralytics (yolov8) if available
            from ultralytics import YOLO as YOLOv8
            self.get_logger().info("Loading YOLOv8n (ultralytics)...")
            # This uses the pre-trained 'yolov8n.pt' that ultralytics provides internally
            self.yolo = YOLOv8('yolov8n.pt')
            self.yolo_type = 'ultralytics'
            self.get_logger().info("YOLOv8 loaded")
        except Exception:
            try:
                import torch
                self.get_logger().info("Loading YOLOv5n via torch.hub (fallback)...")
                self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                self.yolo_type = 'yolov5_hub'
                self.get_logger().info("YOLOv5n (hub) loaded")
            except Exception as e:
                self.get_logger().error(f"Failed loading YOLO: {e}")
                raise

    def run_yolo(self, frame):
        """
        Returns list of objects:
          {"label": str, "bbox":[x1,y1,x2,y2], "center":[cx,cy], "confidence": float}
        """
        if self.yolo is None:
            self.load_yolo()

        objs = []
        try:
            if getattr(self, 'yolo_type', '') == 'ultralytics':
                # ultralytics YOLOv8 API
                results = self.yolo(frame, imgsz=max(self.width, self.height), conf=self.conf_threshold)
                r = results[0]
                # r.boxes: access xyxy, conf, cls
                boxes = getattr(r, 'boxes', None)
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clsids = boxes.cls.cpu().numpy().astype(int)
                else:
                    xyxy = np.array([])
                    confs = np.array([])
                    clsids = np.array([])

                for i, b in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, b[:4])
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    clsid = int(clsids[i]) if i < len(clsids) else 0
                    label = self.yolo.model.names.get(clsid, str(clsid)) if hasattr(self.yolo, 'model') else str(clsid)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    objs.append({"label": label, "bbox":[x1,y1,x2,y2], "center":[cx,cy], "confidence": conf})
            else:
                # torch hub yolov5
                results = self.yolo(frame)  # returns a results object
                # results.xyxy[0] matrix
                xy = results.xyxy[0].cpu().numpy() if hasattr(results.xyxy[0], 'cpu') else np.array(results.xyxy[0])
                for row in xy:
                    x1, y1, x2, y2, conf, clsid = row[:6]
                    x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                    label = self.yolo.names[int(clsid)]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    objs.append({"label": label, "bbox":[x1,y1,x2,y2], "center":[cx,cy], "confidence": float(conf)})
        except Exception as e:
            self.get_logger().error(f"YOLO runtime error: {e}\n{traceback.format_exc()}")

        # keep top-k by confidence
        objs = sorted(objs, key=lambda o: o['confidence'], reverse=True)[:self.max_objects]
        return objs

    # ---------------- BLIP (caption) ----------------
    def load_blip(self):
        if self.blip is not None and self.blip_processor is not None:
            return
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.get_logger().info("Loading BLIP model (may take memory/time)...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.get_logger().info("BLIP loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load BLIP: {e}")
            raise

    def run_blip(self, frame):
        if self.blip is None:
            self.load_blip()
        # frame is BGR; convert to PIL RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        try:
            inputs = self.blip_processor(pil, return_tensors="pt")
            out = self.blip.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            self.get_logger().error(f"BLIP run error: {e}\n{traceback.format_exc()}")
            return ""

    # ---------------- Service callback ----------------
    def analyze_cb(self, request, response):
        """
        Request: description (bool), coordinates (bool)
        Response.message: JSON string with keys:
           description (str) if requested,
           objects (list) if requested
        """
        start = time.time()
        if not request.description and not request.coordinates:
            response.success = False
            response.message = json.dumps({"error":"At least one of description or coordinates must be true"})
            return response

        frame = self.capture_frame()
        if frame is None:
            response.success = False
            response.message = json.dumps({"error":"Failed to capture frame from camera"})
            return response

        out = {}
        # Run YOLO first if coordinates requested (fast)
        if request.coordinates:
            objs = self.run_yolo(frame)
            out['objects'] = objs

        # Run BLIP only if requested (heavy)
        if request.description:
            caption = self.run_blip(frame)
            out['description'] = caption

        # Add metadata
        out['meta'] = {
            "camera_source": self.camera_source,
            "width": self.width,
            "height": self.height,
            "took_s": round(time.time() - start, 3)
        }

        response.success = True
        response.message = json.dumps(out)
        self.get_logger().info(f"analyze_scene -> {json.dumps(out)}")
        return response

    def destroy_node(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SceneService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

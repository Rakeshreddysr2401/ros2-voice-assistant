import json
import logging
from langchain_core.tools import tool
from typing import Dict, Any, List

log = logging.getLogger("SpatialTool")

# Camera resolution constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CENTER_X = FRAME_WIDTH / 2

# Navigation constants (can be tuned)
CENTERING_THRESHOLD = 50  # pixels from center
TARGET_AREA_PERCENT = 0.25 # If object takes up 25% of frame, we are "near"
FOV_DEGREES = 60          # Approximate FOV of the camera

@tool
def spatial_navigator_tool(detections_json: str, target_label: str) -> str:
    """
    Analyzes YOLO detections to provide specific movement instructions to reach a target.
    detections_json: The JSON string returned by describe_objects tool.
    target_label: The name of the object to go near (e.g., 'bottle').
    
    Returns a suggestion for move_robo tool (direction and value).
    """
    try:
        # The describe_objects tool returns a string representation of a list of dicts
        # We need to safely parse it. It might be wrapped in single quotes or be a raw string.
        detections = eval(detections_json) 
        if not isinstance(detections, list):
            return f"Error: Invalid detections format. Expected list, got {type(detections)}"
    except Exception as e:
        return f"Error parsing detections: {str(e)}"

    # Find the target object (pick the most confident one)
    target = None
    max_conf = -1
    for d in detections:
        if d['label'].lower() == target_label.lower():
            if d['confidence'] > max_conf:
                max_conf = d['confidence']
                target = d

    if not target:
        return f"I don't see any '{target_label}' in my current view."

    bbox = target['bbox'] # [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    obj_center_x = (x1 + x2) / 2
    obj_width = x2 - x1
    obj_height = y2 - y1
    obj_area = obj_width * obj_height
    total_area = FRAME_WIDTH * FRAME_HEIGHT
    area_ratio = obj_area / total_area

    # 1. Centering Logic (Turn)
    offset_x = obj_center_x - FRAME_CENTER_X
    
    if abs(offset_x) > CENTERING_THRESHOLD:
        # Calculate approximate degrees to turn
        # Simple linear mapping: pixel_offset / total_width * FOV
        turn_deg = abs(offset_x / FRAME_WIDTH * FOV_DEGREES)
        direction = "R" if offset_x > 0 else "L"
        return f"The {target_label} is not centered. Suggested Action: move_robo(direction='{direction}', value={turn_deg:.1f}) to center it."

    # 2. Distance Logic (Move Forward)
    if area_ratio < TARGET_AREA_PERCENT:
        # Object is still small/far. Move forward.
        # We can scale the movement value based on how small it is
        # If it's very small (e.g. 0.05 area ratio), move 50cm. If it's 0.2, move 10cm.
        dist_cm = max(10, (TARGET_AREA_PERCENT - area_ratio) * 200) 
        return f"The {target_label} is centered but far away (area ratio {area_ratio:.2%}). Suggested Action: move_robo(direction='F', value={dist_cm:.1f}) to get closer."

    return f"I have arrived near the {target_label}. It occupies {area_ratio:.2%} of my view."

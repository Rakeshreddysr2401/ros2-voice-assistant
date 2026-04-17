# 🤖 Humanoid Assistant Project Documentation

This document provides a comprehensive technical overview of the ROS2 Voice Assistant system, detailing its architecture, components, and operational guidelines.

---

## 1. System Architecture
The assistant uses a **Hybrid Multi-Agent System** powered by **LangGraph**. It bridges high-level AI reasoning with low-level robotic control.

### Core Pipeline
*   **Ears (STT)**: `wakeword_node` (Vosk) listens for "Jarvis" to trigger `input_node` (Whisper).
*   **Brain (Agent)**: `agent_node` processes natural language and orchestrates subagents.
*   **Eyes (Vision)**: 
    *   **Fast**: `moondream_server` (<1s feedback for navigation).
    *   **Detailed**: `qwen_vision_server` (Complex reasoning and OCR).
    *   **Object Detection**: `yolo_server` (Bounding box coordinates).
*   **Memory**: 
    *   **Facts**: `Mem0` stores structured user/environment data.
    *   **Space**: `semantic_map_node` tracks object coordinates in 2D space.
*   **Voice (TTS)**: `kokoro_output_node` generates high-quality audio.
*   **Action (Actuators)**: `visual_servoing_node` and `servo_publisher` handle physical movement.

---

## 2. Node Directory

| Node | Executable | Primary Responsibility |
| :--- | :--- | :--- |
| **Wake Word** | `wakeword_node` | Listens for the trigger word ("Jarvis") using local Vosk. |
| **Voice Input** | `input_node` | Transcribes speech to text after wake-up. |
| **Brain** | `agent_node` | Central LangGraph brain coordinating all tools and subagents. |
| **Dashboard** | `dashboard_node` | Serves MJPEG stream and "Thinking" log at `http://localhost:8080`. |
| **Visual Servoing** | `visual_servoing_node` | Continuous object tracking and autonomous search logic. |
| **Semantic Map** | `semantic_map_node` | Dead-reckoning based 2D mapping and object pinning. |
| **Output** | `output_node` | Speech generation using Kokoro TTS. |

---

## 3. Agent Capabilities (Toolbox)

### Navigation & Movement
- `track_object(target)`: Begin continuous visual tracking to follow/reach an object.
- `move_robo(direction, value)`: Execute discrete movements (Forward, Turn, etc.).
- `servo_tool(angle)`: Control robotic arm servos.

### Vision & Perception
- `fast_vision_tool(query)`: High-speed visual check using Moondream.
- `qwen_vision_tool(query)`: Detailed visual analysis using Qwen2.5-VL.
- `describe_objects()`: Get raw YOLO detections with bounding box coordinates.

### Memory & Mapping
- `pin_object_on_map(label)`: Save the current location of an object to the 2D map.
- `query_semantic_map()`: Recall object locations relative to current position.
- `add_memory_tool(fact)`: Store a structured fact in long-term memory (Mem0).
- `search_memory_tool(query)`: Retrieve facts from long-term memory.

### Transparency
- `status_tool(message)`: Update the web dashboard with the robot's current thought or intent.

---

## 4. Setup & Installation

### Dependencies
1.  **AI Models (Ollama)**:
    - `ollama pull moondream`
    - `ollama pull qwen2.5-vl`
2.  **Speech Model**:
    - Download `vosk-model-small-en-us` and place in `voice_assistant_pkg/voice_assistant_pkg/model-small`.
3.  **Python Packages**:
    ```bash
    pip install mem0ai faster-whisper sounddevice vosk opencv-python rclpy
    ```

### Running the System
1.  **Build**: `colcon build --packages-select voice_assistant_pkg`
2.  **Launch Core**: Run `wakeword_node`, `input_node`, `agent_node`, and `output_node`.
3.  **Launch Vision**: Run `yolo_server`, `moondream_server`, and `visual_servoing_node`.
4.  **Launch Visualization**: Run `dashboard_node` and open `http://localhost:8080`.

---

## 5. Operational Workflow Example

1.  **Trigger**: User says "Jarvis".
2.  **Command**: User says "Go near the bottle."
3.  **Search**: Agent uses `describe_objects` to find the bottle.
4.  **Track**: Agent calls `track_object("bottle")`.
5.  **Move**: `visual_servoing_node` drives the robot while keeping the bottle centered.
6.  **Search Recovery**: If bottle is lost, robot rotates 360° to find it.
7.  **Finalize**: Upon arrival, robot calls `pin_object_on_map("bottle")` and says "I have arrived."

---
**Version**: 1.0.0  
**Status**: Integrated Visual Servoing, Semantic Mapping, and Wake Word.

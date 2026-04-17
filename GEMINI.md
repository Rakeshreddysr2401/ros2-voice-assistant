# ROS2 Voice Assistant - Project Mandates

This document serves as the primary source of truth for development standards, architectural decisions, and configuration management for the ROS2 Voice Assistant project.

## 🤖 Model Selection & Configuration

The assistant uses a multi-model architecture. Models are primarily selected via environment variables.

### 1. Main LLM (Agent)
The main brain of the assistant is controlled via `agent_node.py`.
- **Primary Variable**: `AGENT_MODEL`
  - Default: `openai:gpt-4o-mini`
  - Local (Ollama) Example: `ollama:qwen2.5:7b` (Note: requires `OLLAMA_HOST` if not local)
- **Configuration File**: `src/voice_assistant_pkg/voice_assistant_pkg/llm_config.py`
  - Used by `chatAgentNode.py` and some legacy tools.
  - Can be toggled between `ChatOpenAI` and `ChatOllama`.

### 2. Vision Models
- **Qwen Vision**: Controlled via `qwen_vision_server.py`.
  - Env Var: `MODEL` (Default: `qwen2.5vl:3b`)
  - Env Var: `OLLAMA_HOST` (Default: `192.168.1.22`)
- **YOLO**: Controlled via `yolo_server.py`.
  - Currently hardcoded to `yolov8s.pt`.
- **BLIP**: Controlled via `blip_server.py`.

### 3. Speech (TTS)
- **Kokoro**: Controlled via `kokoro_output_node.py`.
  - Model: `kokoro-v1.0.int8.onnx`
  - Voice: `af_sarah` (Default), can be changed in the node's `__init__`.

---

## ⚙️ Environment Variables

Ensure these are set in your shell or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Required for OpenAI models | - |
| `TAVILY_API_KEY` | Required for web search tool | - |
| `AGENT_MODEL` | Main LLM model string | `openai:gpt-4o-mini` |
| `OLLAMA_HOST` | IP/Host for local Ollama server | `192.168.1.22` |
| `QDRANT_URL` | URL for Qdrant knowledge base | - |
| `QDRANT_API_KEY` | API Key for Qdrant | - |

---

## 🏗️ System Architecture

1. **Input**: `input_node` (STT) publishes to `user_input`.
2. **Brain**: `agent_node` (LangGraph) processes input, calls tools, and decides on actions.
3. **Vision**: `qwen_vision_server`, `yolo_server`, and `blip_server` provide visual feedback via services.
4. **Action**: `servo_publisher` and `micro_ros_agent` handle hardware (ESP32) movement.
5. **Output**: `output_node` (Kokoro TTS) speaks back to the user.

---

## 🛠️ Development Standards

- **ROS2 Conventions**: Always use `rclpy` and follow standard ROS2 node patterns.
- **Service Interfaces**: All custom services are defined in `custom_interfaces/`.
- **Tooling**: New agent capabilities should be added as tools in `voice_assistant_pkg/tools/`.
- **Verification**:
  - For hardware changes: Test with `ros2 topic pub /movement_cmd`.
  - For agent changes: Verify tool calling logs in `agent_node` output.
  - For vision changes: Test services via `ros2 service call`.

## 🚀 Common Execution Commands

```bash
# Build
colcon build --packages-select voice_assistant_pkg custom_interfaces

# Run core nodes
ros2 run voice_assistant_pkg input_node
ros2 run voice_assistant_pkg agent_node
ros2 run voice_assistant_pkg output_node

# Run vision servers
ros2 run voice_assistant_pkg qwen_vision_server
ros2 run voice_assistant_pkg yolo_server
```

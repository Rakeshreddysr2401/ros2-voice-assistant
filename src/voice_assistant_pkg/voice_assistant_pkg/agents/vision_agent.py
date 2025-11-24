from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from .base_agent import BaseAgent
from ..tools import get_vision_tools


class VisionAgent(BaseAgent):
    def __init__(self):
        super().__init__("VisionAgent", get_vision_tools())

    def build_system_message(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "You are the robot's VISION specialist.\n"
                "- Use your tools to describe what the camera sees, "
                "answer VQA queries, and locate objects.\n"
                "- Prefer qwen_vision_tool for rich scene description.\n"
                "- Prefer describe_objects (YOLO) when user cares about "
                "object names, positions, or coordinates.\n"
            )
        )


VISION_AGENT = VisionAgent()


@tool
def vision_agent_tool(query: str) -> str:
    """
    High-level Vision Agent.

    Use this when the user asks about what the robot can see, objects,
    their positions, or wants a description of the scene.
    """
    print(f"[VisionAgent] Handling query: {query}")
    return VISION_AGENT.run_query(query)

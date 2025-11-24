from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from .base_agent import BaseAgent
from ..tools import get_movement_tools


class MovementAgent(BaseAgent):
    def __init__(self):
        super().__init__("MovementAgent", get_movement_tools())

    def build_system_message(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "You control the robot's physical MOVEMENT and lights.\n"
                "- You have a single servo arm with angles 0–180 degrees.\n"
                "  * 0° = fully down, 90° = horizontal, 180° = fully up.\n"
                "- You can also set traffic lights: red, green, orange.\n"
                "- Always\n"
                "  * Clamp angles to [0, 180].\n"
                "  * Confirm actions in natural language.\n"
            )
        )


MOVEMENT_AGENT = MovementAgent()


@tool
def movement_agent_tool(command: str) -> str:
    """
    High-level Movement Agent.

    Use this for commands like raising/lowering the hand, moving to a
    specific angle, or controlling the traffic lights.
    """
    print(f"[MovementAgent] Handling command: {command}")
    return MOVEMENT_AGENT.run_query(command)

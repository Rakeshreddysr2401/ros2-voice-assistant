#tools.__init__.py
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool, StructuredTool

from .qdrant_tool import qdrant_search_tool
from .tavily_tool import tavily_tool
from .yolo_tool import describe_objects
from .qwen_tool import qwen_vision_tool
from .lt import set_traffic_light
from .servo_tool import move_servos


def wrap_tool(func):
    """Wrap a function as a Tool, ensuring it has a name and description."""
    if isinstance(func, (Tool, StructuredTool)):
        return func

    return Tool.from_function(
        func=func,
        name=getattr(func, "__name__", "unnamed_tool"),
        description=func.__doc__ or "No description provided.",
    )


# ---------- Low-level tool groups ----------

def get_vision_tools():
    """Tools used by the VisionAgent."""
    return [
        wrap_tool(qwen_vision_tool),
        wrap_tool(describe_objects),
        # add blip or other vision tools here
    ]


def get_movement_tools():
    """Tools used by the MovementAgent."""
    return [
        wrap_tool(move_servos),
        wrap_tool(set_traffic_light),  # if you want lights as part of 'movement'
    ]


def get_knowledge_tools():
    """Tools used by the Knowledge/Reasoning Agent."""
    return [
        wrap_tool(qdrant_search_tool),
        wrap_tool(tavily_tool),
    ]


def get_all_primitive_tools():
    """Convenience helper: all low-level tools in one list (optional)."""
    return (
        get_vision_tools()
        + get_movement_tools()
        + get_knowledge_tools()
    )


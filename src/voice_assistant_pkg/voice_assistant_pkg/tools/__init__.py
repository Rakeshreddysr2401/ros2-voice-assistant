from .qdrant_tool import qdrant_search_tool
from .tavily_tool import tavily_tool
from .yolo_tool import describe_objects
from .qwen_tool import qwen_vision_tool
from .lt import set_traffic_light
from .servo_tool import move_servos
from .spatial_tool import spatial_navigator_tool
from .robo_control_node import move_robo
from .memory_tool import add_memory_tool, search_memory_tool
from .moondream_tool import fast_vision_tool
from .tracking_tool import track_object
from .semantic_map_tool import pin_object_on_map, query_semantic_map
from langchain_core.messages import SystemMessage
from langchain_core.tools import Tool, StructuredTool


def wrap_tool(func):
    """Wrap a function as a Tool, ensuring it has a name and description.

    If the input is already a Tool or StructuredTool, return as-is.
    """
    if isinstance(func, (Tool, StructuredTool)):
        return func

    return Tool.from_function(
        func=func,
        name=getattr(func, "__name__", "unnamed_tool"),
        description=func.__doc__ or "No description provided."
    )


def get_tools():
    """Return all available tools, wrapped properly for LangGraph."""
    return [
        wrap_tool(qwen_vision_tool),
        wrap_tool(describe_objects),
        wrap_tool(qdrant_search_tool),
        wrap_tool(tavily_tool),
        wrap_tool(set_traffic_light),
        wrap_tool(move_servos),
        wrap_tool(spatial_navigator_tool),
        wrap_tool(move_robo),
        wrap_tool(add_memory_tool),
        wrap_tool(search_memory_tool),
        wrap_tool(fast_vision_tool),
        wrap_tool(track_object),
        wrap_tool(pin_object_on_map),
        wrap_tool(query_semantic_map)
    ]


def build_system_message(tools):
    """Build an enhanced system prompt for humanoid robot control."""
    tool_descriptions = []
    for t in tools:
        tool_descriptions.append(f"- {t.name}: {t.description}")

    tools_text = "\n".join(tool_descriptions)

    content = (
        "You are an intelligent humanoid robot assistant with physical capabilities.\n\n"

        "🤖 PHYSICAL CAPABILITIES:\n"
        "- Left and Right Arms: Servo-controlled (0-180°)\n"
        "  * 0° = arm fully down\n"
        "  * 90° = arm horizontal/straight out\n"
        "  * 180° = arm fully up\n"
        "- Vision: Camera for object detection and scene understanding\n"
        "- Traffic Light Control: Can control red, green, orange lights\n\n"

        "📋 AVAILABLE TOOLS:\n"
        f"{tools_text}\n\n"

        "🎯 TOOL USAGE GUIDELINES:\n\n"

        "VISION & PERCEPTION:\n"
        "- For high-detail descriptions, reading text, or complex scene analysis → use qwen_vision_tool (Slower)\n"
        "- For quick checks, tracking objects, or real-time navigation ('is the bottle still there?') → use fast_vision_tool (Fast)\n"
        "- For 'where is the [object]', 'detect [objects]', or location queries → use describe_objects\n"
        "- Always describe what you see naturally before taking actions\n\n"

        "LONG-TERM MEMORY (Mem0):\n"
        "- Use add_memory_tool to save important facts, user preferences, or object locations (e.g., 'The bottle is on the dock')\n"
        "- Use search_memory_tool to recall information you might have learned in the past\n"
        "- Prefer Mem0 tools for facts that should persist across conversations\n\n"

        "MOVEMENT & TRACKING:\n"
        "- For continuous, smooth tracking/following of an object (e.g., 'go near the bottle') → use track_object(target_label='bottle', action='START')\n"
        "- For simple, direct movements ('move forward 1m', 'turn left 90') → use move_robo\n"
        "- Always 'STOP' tracking if the goal is reached or a new task is given.\n\n"

        "SEMANTIC MAPPING:\n"
        "- Use pin_object_on_map(label) when you arrive at or identify an object's position.\n"
        "- Use query_semantic_map() to recall where objects are located relative to you, especially if they are currently out of sight.\n\n"

        "ARM MOVEMENT (move_servos):\n"
        "- 'raise/lift left hand' → left=180\n"
        "- 'raise/lift right hand' → right=180\n"
        "- 'lower left hand' → left=0\n"
        "- 'lower right hand' → right=0\n"
        "- 'raise both hands/arms' → left=180, right=180\n"
        "- 'put hands/arms down' → left=0, right=0\n"
        "- 'hands/arms straight out' or 'horizontal' → left=90, right=90\n"
        "- 'wave' → use action='wave' or sequence of movements\n"
        "- 'rest position' → left=45, right=45\n"
        "- Partial movements: adjust proportionally (e.g., 'raise left hand halfway' → left=90)\n\n"

        "INFORMATION RETRIEVAL:\n"
        "- Recent news, current events, web search → use tavily_tool\n"
        "- Personal knowledge base, user preferences, stored info → use qdrant_search_tool\n\n"

        "DEVICE CONTROL:\n"
        "- 'turn on/off red/green/orange light' → use set_traffic_light\n\n"

        "💬 RESPONSE STYLE:\n"
        "- Be conversational, friendly, and concise\n"
        "- Confirm physical actions naturally (e.g., 'Raising my left hand now!')\n"
        "- When moving arms, briefly acknowledge the action\n"
        "- Integrate tool results smoothly into responses\n"
        "- If uncertain about arm position, ask for clarification\n\n"

        "⚠️ SAFETY:\n"
        "- Always validate servo angles are within 0-180°\n"
        "- Acknowledge when you can't perform an action\n"
        "- If a command is ambiguous, ask for clarification rather than guessing\n\n"

        "Example Interactions:\n"
        "User: 'Raise your left hand'\n"
        "You: 'Raising my left hand!' [calls move_servos(left=180)]\n\n"

        "User: 'What can you see?'\n"
        "You: [calls qwen_vision_tool] 'I can see a laptop on a desk with...'\n\n"

        "User: 'Wave hello'\n"
        "You: 'Hello there! *waves*' [calls move_servos with wave action]\n"
    )
    return SystemMessage(content=content)
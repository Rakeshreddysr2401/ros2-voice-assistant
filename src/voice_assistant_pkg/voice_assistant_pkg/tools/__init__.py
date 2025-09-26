# tools/__init__.py
from .qdrant_tool import qdrant_search_tool
from .tavily_tool import tavily_tool
from .blip_tool import describe_scene
from .yolo_tool import describe_objects
from .ollama import ollama_query
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
        wrap_tool(describe_scene),
        wrap_tool(describe_objects),
        wrap_tool(qdrant_search_tool),
        wrap_tool(tavily_tool),
        wrap_tool(ollama_query)
    ]


def build_system_message(tools):
    """Build a system prompt describing available tools and when to use them."""
    tool_descriptions = []
    for t in tools:
        # StructuredTool and Tool have .name and .description attributes
        tool_descriptions.append(f"- {t.name}: {t.description}")

    tools_text = "\n".join(tool_descriptions)

    content = (
        "You are Rakesh's AI assistant with access to camera vision and search capabilities.\n\n"
        "Available tools:\n"
        f"{tools_text}\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- When asked about what you can see, objects in view, or visual questions, ALWAYS use describe_scene first to get image description\n"
        "- When asked about specific objects, their locations, or detection tasks, use describe_objects to get object coordinates\n"
        "- For questions requiring recent information or web search, use tavily_tool\n"
        "- For knowledge base queries, use qdrant_search_tool\n"
        # "- Combine multiple tools when needed (e.g., first get scene description, then object details)\n"
        "- Always provide context about what the camera is seeing when answering visual questions\n\n"
        "Respond naturally and conversationally, integrating tool results smoothly into your answers."
    )
    return SystemMessage(content=content)

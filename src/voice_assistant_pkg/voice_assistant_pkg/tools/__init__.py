#tools __init__.py
from .qdrant_tool import qdrant_search_tool
from .tavily_tool import tavily_tool
from .blip_tool import describe_scene
from .yolo_tool import describe_objects
from .ollama import ollama_query
from langchain_core.messages import SystemMessage

def get_tools():
    """Return all available tools."""
    return [describe_scene, describe_objects, qdrant_search_tool, tavily_tool,ollama_query]


def build_system_message(tools):
    """Build a system prompt describing available tools and when to use them."""
    tool_descriptions = []
    for t in tools:
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
        "- Combine multiple tools when needed (e.g., first get scene description, then object details)\n"
        "- Always provide context about what the camera is seeing when answering visual questions\n\n"
        "Respond naturally and conversationally, integrating tool results smoothly into your answers."
    )
    return SystemMessage(content=content)

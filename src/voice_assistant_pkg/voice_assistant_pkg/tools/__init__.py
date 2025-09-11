from .qdrant_tool import qdrant_search_tool
from .tavily_tool import tavily_tool
from .blip_tool import describe_scene
from langchain_core.messages import SystemMessage

def get_tools():
    """Return all available tools."""
    return [qdrant_search_tool, tavily_tool,describe_scene]

def build_system_message(tools):
    """Build a system prompt describing available tools."""
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    content = (
        "You are Rakesh’s AI assistant.\n\n"
        "You can use the following tools when needed:\n"
        f"{tool_descriptions}\n\n"
        "If you are unsure or need more information, always check if a tool can help.\n"
        "Use multiple tools sequentially if needed, and combine their results in your response."

    )
    return SystemMessage(content=content)

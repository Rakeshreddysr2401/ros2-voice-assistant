#tools __init__.py
from .qdrant_retriever import qdrant_search_tool
from .tavily_tool import tavily_tool

def get_tools():
    """Return all available tools as a list."""
    return [tavily_tool, qdrant_search_tool]


def build_system_message(tools):
    """Build system message dynamically from available tools."""
    tool_list = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    from langchain_core.messages import SystemMessage
    return SystemMessage(
        content=f"""
You are a helpful AI assistant. You have access to the following tools:

{tool_list}

Whenever a user query requires external information, choose the most appropriate tool.
Format tool calls exactly as JSON like:

{{"tool": "<tool_name>", "input": "<user query>"}}

If no tool is needed, respond directly.
Always be helpful, clear, and concise.
"""
    )

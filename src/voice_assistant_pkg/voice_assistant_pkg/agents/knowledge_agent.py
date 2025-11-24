from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from .base_agent import BaseAgent
from ..tools import get_knowledge_tools


class KnowledgeAgent(BaseAgent):
    def __init__(self):
        super().__init__("KnowledgeAgent", get_knowledge_tools())

    def build_system_message(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "You are the robot's KNOWLEDGE specialist.\n"
                "- Use qdrant_search_tool for personal or project-related "
                "information about the user or Rakesh.\n"
                "- Use tavily_tool for general web search, news, weather, etc.\n"
                "- Combine both if needed, and respond with a clear summary.\n"
            )
        )


KNOWLEDGE_AGENT = KnowledgeAgent()


@tool
def knowledge_agent_tool(query: str) -> str:
    """
    High-level Knowledge/Research Agent.

    Use this for questions that need personal knowledge base lookup (Qdrant)
    or general web search (Tavily).
    """
    print(f"[KnowledgeAgent] Handling query: {query}")
    return KNOWLEDGE_AGENT.run_query(query)

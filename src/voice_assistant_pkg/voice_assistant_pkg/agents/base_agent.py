from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from ..llm_config import llm  # your base ChatOpenAI instance


class BaseAgent(ABC):
    """Common helper for sub-agents."""

    def __init__(self, name: str, tools: List[BaseTool]):
        self.name = name
        self.tools = tools
        # Bind tools to this agent's LLM
        self.llm = llm.bind_tools(self.tools)

    @abstractmethod
    def build_system_message(self) -> SystemMessage:
        """Each agent provides its own role description."""
        ...

    def run_query(self, query: str) -> str:
        """Run a fresh 1-shot query to this agent."""
        messages: List[BaseMessage] = [
            self.build_system_message(),
            HumanMessage(content=query),
        ]
        response = self.llm.invoke(messages)

        # If this agent itself used tools, LLM already called them via tool-calling
        return getattr(response, "content", str(response))

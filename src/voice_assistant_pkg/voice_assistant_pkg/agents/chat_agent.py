from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import BaseTool

from ..states.states import AgentState
from ..llm_config import llm
from .vision_agent import vision_agent_tool
from .movement_agent import movement_agent_tool
from .knowledge_agent import knowledge_agent_tool


def build_parent_system_message(tools: List[BaseTool]) -> SystemMessage:
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools
    )

    content = (
        "You are the main CHAT AGENT for a humanoid robot.\n"
        "- You talk with the human and decide which specialist agent to call.\n\n"
        "Specialist agents (exposed as tools):\n"
        f"{tool_descriptions}\n\n"
        "Routing guidance:\n"
        "- Vision / camera / what you see / objects / coordinates → use vision_agent_tool.\n"
        "- Movement / servo / raise hand / lights → use movement_agent_tool.\n"
        "- Personal data, skills, projects, or general web info → use knowledge_agent_tool.\n"
        "- You may also answer simple chitchat yourself without tools.\n"
        "- You can call tools multiple times if needed and then summarise.\n"
    )
    return SystemMessage(content=content)


# Tools representing sub-agents
PARENT_TOOLS: List[BaseTool] = [
    vision_agent_tool,
    movement_agent_tool,
    knowledge_agent_tool,
]

# LLM bound to sub-agent tools
PARENT_LLM = llm.bind_tools(PARENT_TOOLS)


def call_parent_agent(state: AgentState) -> Dict[str, Any]:
    """Node function used in LangGraph: parent chat + routing."""
    try:
        messages: List[BaseMessage] = state["messages"]

        system_msg = build_parent_system_message(PARENT_TOOLS)

        if not messages or not isinstance(messages[0], SystemMessage):
            conversation = [system_msg] + messages
        else:
            # Keep existing system if you already inserted it
            conversation = messages

        print("[ChatAgent] Invoking parent LLM with sub-agent tools...")
        response = PARENT_LLM.invoke(conversation)

        # You can inspect response.tool_calls here if you like
        if getattr(response, "tool_calls", None):
            tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
            print(f"[ChatAgent] LLM requested sub-agents: {tool_names}")

        return {"messages": messages + [response]}

    except Exception as e:
        from langchain_core.messages import AIMessage

        print(f"[ChatAgent] Error: {e}")
        error_msg = AIMessage(
            content=(
                "I encountered an internal error while handling your request. "
                f"Details: {e}"
            )
        )
        return {"messages": state.get("messages", []) + [error_msg]}

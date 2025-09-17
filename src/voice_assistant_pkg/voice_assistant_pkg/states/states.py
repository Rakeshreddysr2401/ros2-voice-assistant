from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ------------------ State Definition ------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    chatAgentResponse: Optional[AIMessage]
    users_query: Optional[str]
    retry_count: int
    final_response: Optional[AIMessage]
    review_feedback: Optional[dict]
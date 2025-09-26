from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ------------------ State Definition ------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    users_query: Optional[str]
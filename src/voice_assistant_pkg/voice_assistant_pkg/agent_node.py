#!/usr/bin/env python3
"""
agent_node.py — DeepAgents ROS2 Node (Clean + Qdrant FIXED)
- ONLY Qdrant part updated
- Added logs
- Replaced deprecated class
- Everything else unchanged exactly as you requested
"""

import os
import logging
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from deepagents import create_deep_agent
from deepagents.backends import StateBackend

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_core.messages import SystemMessage
from langchain.tools import tool

from tavily import TavilyClient

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings


# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AgentNode")


# ---------------- ENV ----------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL","https://14913e1e-77ca-4f9d-bf4b-26bf8d4f4230.eu-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.z57EvchkzSJuTD-b3Rx4za-mA20RNBHhZ9d-g9As8HY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_knowledge_base")
MODEL_NAME_EMBED = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
AGENT_MODEL  = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")


# ==================================================================
#  FIXED TOOLS — PLAIN FUNCTIONS
# ==================================================================

_shared_agent_node = None   # global pointer used by speak tool


@tool
def speak_tool(message: str) -> str:
    """
    Publish text to /agent_response → OutputNode (TTS).
    LLM must ALWAYS use this to talk to the user.
    """
    global _shared_agent_node
    if _shared_agent_node is None:
        return "[error: AgentNode not ready]"

    msg = String()
    msg.data = message
    _shared_agent_node.pub_response.publish(msg)

    log.info(f"[TOOL:speak_tool] Published: {message}")
    return f"[spoken] {message}"


# ---------------- Tavily setup ----------------
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    tavily_client = None


@tool
def tavily_tool(query: str, max_results: int = 5) -> str:
    """
    Tavily web search.
    """
    log.info(f"[TOOL:tavily_tool] Query={query}")

    if tavily_client is None:
        return "Tavily not configured."

    try:
        res = tavily_client.search(query=query, max_results=max_results)
        results = res.get("results", [])
        if not results:
            return "No Tavily results."

        text = "\n".join(
            f"- {r.get('title')} | {r.get('url')}\n  {r.get('content')[:200]}"
            for r in results
        )
        return text

    except Exception as e:
        return f"Tavily error: {str(e)}"


# ---------------- Qdrant FIXED SETUP ----------------
try:
    if QDRANT_URL and QDRANT_API_KEY:
        log.info(f"[QDRANT] Connecting to: {QDRANT_URL}")

        q_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
            check_compatibility=False
        )

        # Check collections
        try:
            info = q_client.get_collections()
            log.info(f"[QDRANT] Collections on server: {info}")
            existing = [c.name for c in info.collections]
            if QDRANT_COLLECTION not in existing:
                log.warning(f"[QDRANT] Collection '{QDRANT_COLLECTION}' does NOT exist!")
        except Exception as e:
            log.error(f"[QDRANT] Failed to get collections: {e}")

        q_emb = OpenAIEmbeddings(model=MODEL_NAME_EMBED)

        q_vectorstore = QdrantVectorStore(
            client=q_client,
            collection_name=QDRANT_COLLECTION,
            embedding=q_emb
        )

        log.info("[INIT] QdrantVectorStore ready.")

    else:
        q_vectorstore = None
        log.error("[QDRANT] URL or API Key missing.")

except Exception as e:
    log.error(f"[QDRANT INIT ERROR] {str(e)}")
    q_vectorstore = None


@tool
def qdrant_search_tool(query: str, top_k: int = QDRANT_TOP_K) -> str:
    """
    Search personal knowledge base in Qdrant.
    """
    log.info(f"[TOOL:qdrant_search_tool] Query={query}")

    if q_vectorstore is None:
        log.error("[QDRANT] Vectorstore is None. Memory disabled.")
        return "Memory system not configured."

    try:
        log.info(f"[QDRANT] Searching in collection '{QDRANT_COLLECTION}' top_k={top_k}")

        matches = q_vectorstore.similarity_search(query, k=top_k)

        log.info(f"[QDRANT] Matches found: {len(matches)}")

        for i, doc in enumerate(matches):
            log.info(f"[QDRANT] {i} => {doc.page_content} | metadata={doc.metadata}")

        if not matches:
            return "No matching knowledge found."

        return "\n".join(
            f"- {doc.page_content} (source={doc.metadata.get('source','unknown')})"
            for doc in matches
        )

    except Exception as e:
        log.error(f"[QDRANT ERROR] {e}")
        return f"Qdrant error: {str(e)}"


@tool
def vision_tool(target: str) -> dict:
    """
    Mock vision. Replace with actual vision service later.
    """
    mock = {
        "chair": {"found": True, "x_offset": -0.2, "distance": 0.8},
        "table": {"found": True, "x_offset": 0.1, "distance": 1.5},
    }
    return mock.get(target.lower(), {"found": False})


@tool
def movement_tool(direction: str, amount: float = 0.2) -> str:
    """
    Movement control (Twist publisher).
    """
    global _shared_agent_node
    if _shared_agent_node is None:
        return "[error: AgentNode not ready]"

    from geometry_msgs.msg import Twist
    twist = Twist()

    if direction == "forward":
        twist.linear.x = amount
    elif direction == "backward":
        twist.linear.x = -amount
    elif direction == "left":
        twist.angular.z = +amount
    elif direction == "right":
        twist.angular.z = -amount
    elif direction == "stop":
        pass
    else:
        return f"[error: invalid direction '{direction}']"

    log.info(f"[TOOL:movement_tool] {direction}, {amount}")
    return f"[movement:{direction}]"


# ==================================================================
# ROS2 AGENT NODE
# ==================================================================
class AgentNode(Node):

    def __init__(self):
        super().__init__("agent_node")

        global _shared_agent_node
        _shared_agent_node = self

        # ROS topics
        self.sub_input    = self.create_subscription(String, "user_input", self._on_user_input, 10)
        self.pub_response = self.create_publisher(String, "agent_response", 10)

        # memory
        self.store = InMemoryStore()
        self.checkpointer = MemorySaver()

        # Subagents
        self.subagents = [
            {
                "name": "research-subagent",
                "description": "Web research using Tavily.",
                "system_prompt": (
                    "You perform research using tavily_tool.\n"
                    "When replying to the user, ALWAYS use speak_tool.\n"
                ),
                "tools": [tavily_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "memory-subagent",
                "description": "Knowledge lookup from Qdrant.",
                "system_prompt": (
                    "Use qdrant_search_tool for memory queries.\n"
                    "ALWAYS reply using speak_tool.\n"
                ),
                "tools": [qdrant_search_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "communication-subagent",
                "description": "Handles speaking.",
                "system_prompt": "Always use speak_tool(text) to communicate.",
                "tools": [speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "movement-subagent",
                "description": (
                    "Navigation controller.\n"
                    "Use vision_tool(target) to observe.\n"
                    "Use movement_tool() to move.\n"
                    "ANNOUNCE progress using speak_tool.\n"
                ),
                "system_prompt": (
                    "For movement tasks:\n"
                    "1. Observe environment using vision_tool(target).\n"
                    "2. Decide appropriate movement using movement_tool.\n"
                    "3. ALWAYS communicate using speak_tool.\n"
                ),
                "tools": [vision_tool, movement_tool, speak_tool],
                "model": AGENT_MODEL,
            }
        ]

        # Master system instructions
        self.system_prompt = SystemMessage(
            content=(
                "You are the robot's supervisor.\n"
                "IMPORTANT: The ONLY way you can communicate with the user is using speak_tool(message).\n"
                "NEVER output plain text.\n"
                "Delegate to appropriate subagents.\n"
                "If user requests movement, delegate to movement-subagent.\n"
            )
        )

        # Create DeepAgent
        self.agent = create_deep_agent(
            model=AGENT_MODEL,
            tools=[speak_tool, tavily_tool, qdrant_search_tool],
            subagents=self.subagents,
            system_prompt=self.system_prompt.content,
            backend=lambda rt: StateBackend(rt),
            store=self.store,
            checkpointer=self.checkpointer,
        )

        log.info("AgentNode initialized successfully.")

    # ---------------- user input ----------------
    def _on_user_input(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        log.info(f"\n==== USER SAID: {text} ====")

        config = {"configurable": {"thread_id": "main_conversation"}}

        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                config=config
            )

            self._auto_resume_interrupts(result, config)

        except Exception as e:
            log.error(f"[AGENT ERROR] {str(e)}")
            out = String()
            out.data = f"Error: {str(e)}"
            self.pub_response.publish(out)

    # ---------------- interrupt handling ----------------
    def _auto_resume_interrupts(self, result, config):
        while "__interrupt__" in result:
            intr = result["__interrupt__"][0].value
            decisions = [{"type": "approve"} for _ in intr["action_requests"]]
            result = self.agent.invoke(
                Command(resume={"decisions": decisions}), config=config
            )


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

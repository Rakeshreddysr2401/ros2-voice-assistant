#!/usr/bin/env python3
"""
agent_node.py — DeepAgents ROS2 Node (TOOLS FIXED: no class-method tools)
"""

import os
import logging
from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

from tavily import TavilyClient

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings


# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("AgentNode")


# ---------------- ENV ----------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_knowledge_base")
MODEL_NAME_EMBED = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
AGENT_MODEL  = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")


# ==================================================================
#  FIXED TOOLS — PLAIN FUNCTIONS (NO METHODS!)
# ==================================================================

_shared_agent_node = None   # global pointer used by speak tool


@tool
def speak_tool(message: str) -> str:
    """
    Publish text to /agent_response → OutputNode (TTS).
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
    Perform a Tavily web search and return formatted results.
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


# ---------------- Qdrant setup ----------------
try:
    if QDRANT_URL and QDRANT_API_KEY:
        q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        q_emb = OpenAIEmbeddings(model=MODEL_NAME_EMBED)
        q_vectorstore = Qdrant(client=q_client,
                               collection_name=QDRANT_COLLECTION,
                               embeddings=q_emb)
        log.info("[INIT] Qdrant ready.")
    else:
        q_vectorstore = None
except Exception as e:
    log.error(f"[QDRANT INIT ERROR] {str(e)}")
    q_vectorstore = None


@tool
def qdrant_search_tool(query: str, top_k: int = QDRANT_TOP_K) -> str:
    """
    Search personal knowledge stored in Qdrant.
    """
    log.info(f"[TOOL:qdrant_search_tool] Query={query}")

    if q_vectorstore is None:
        return "Qdrant not configured."

    try:
        matches = q_vectorstore.similarity_search(query, k=top_k)
        if not matches:
            return "No matching knowledge found."

        return "\n".join(
            f"- {doc.page_content} (source={doc.metadata.get('source','unknown')})"
            for doc in matches
        )

    except Exception as e:
        return f"Qdrant error: {str(e)}"


@tool
def vision_tool(target: str) -> dict:
    """
    Returns rough position of an object.
    Output example:
        {"found": True, "x_offset": -0.3, "distance": 1.2}
    """
    # PLACEHOLDER: real version will query a vision node
    mock = {
        "chair": {"found": True, "x_offset": -0.2, "distance": 0.8},
        "table": {"found": True, "x_offset": 0.1, "distance": 1.5},
    }
    return mock.get(target.lower(), {"found": False})


@tool
def movement_tool(direction: str, amount: float = 0.2) -> str:
    """
    Move the robot. Directions: forward, backward, left, right, stop.
    Publishes Twist to /cmd_vel.
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
        twist.linear.x = 0.0
        twist.angular.z = 0.0
    else:
        return f"[error: invalid direction '{direction}']"

    # _shared_agent_node.pub_cmd_vel.publish(twist)
    log.info(f"[TOOL:movement_tool] {direction} amt={amount}")
    return f"[movement] {direction}"



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
                "system_prompt": "Use tavily_tool for research.",
                "tools": [tavily_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "memory-subagent",
                "description": "Qdrant knowledge lookup.",
                "system_prompt": "Use qdrant_search_tool for memory queries.",
                "tools": [qdrant_search_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "communication-subagent",
                "description": "Handles speaking.",
                "system_prompt": "Always use speak_tool(text).",
                "tools": [speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "movement-subagent",
                "description": (
                    "Autonomous navigation subagent. "
                    "Given a target object like `chair`, repeatedly use vision_tool to "
                    "determine relative direction and distance, then call movement_tool "
                    "to adjust left/right/forward. Continue until the object is centered "
                    "and distance < 0.3m."
                ),
                "system_prompt": (
                    "You control robot movement.\n"
                    "Loop:\n"
                    "1. Ask vision_tool(target) to get object offset/distance.\n"
                    "2. If x_offset < -0.1 → movement_tool('left').\n"
                    "3. If x_offset > +0.1 → movement_tool('right').\n"
                    "4. If distance > 0.4 → movement_tool('forward').\n"
                    "5. If distance < 0.3 → movement_tool('stop') and exit.\n"
                    "6. Repeat.\n"
                    "Always use speak_tool to announce progress to the user.\n"
                ),
                "tools": [vision_tool, movement_tool, speak_tool],
                "model": AGENT_MODEL,
            }

            ##movement agent (used to move left right forward backward) - It Needs to have capability like if it receives coomand to go near chair
            ##then it need to take decision to move forward or left or right based on chair position continuously until it reaches its goal it can use other subagents if needed
            ## like for vision or any queries to take next decision
        ]

        self.system_prompt = SystemMessage(
            content=(
                "You are the robot's supervisor.\n"
                "For ANY user-facing message, use speak_tool(message).\n"
                "Use subagents when appropriate.\n"
                "If the user asks to move somewhere or approach an object, "
                "delegate to movement-subagent.\n"
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

        log.info("AgentNode initialized.")

    # ---------------- user input ----------------
    def _on_user_input(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        log.info(f"\n===== USER SAID: {text} =====")

        config = {"configurable": {"thread_id": "main_conversation"}}

        try:
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": text}]},
                config=config
            )

            # process
            result = self._auto_resume_interrupts(result, config)
            self._fallback_speak(result)

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
        return result

    # ---------------- fallback if LLM forgets speak_tool ----
    def _fallback_speak(self, result):
        for m in result.get("messages", []):
            content = getattr(m, "content", "")
            if content and not content.startswith("[spoken]"):
                msg = String()
                msg.data = content
                self.pub_response.publish(msg)
                log.info(f"[FallbackSpeak] {content}")


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

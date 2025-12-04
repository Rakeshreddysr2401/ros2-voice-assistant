#!/usr/bin/env python3
"""
agent_node.py — DeepAgents ROS2 Node (Clean + Qdrant + Vision Subagent)
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


# -------------------------------------------------------
#  IMPORTING YOUR VISION TOOLS
# -------------------------------------------------------
from .tools.yolo_tool import describe_objects        # YOLO
from .tools.blip_tool import describe_scene          # BLIP
from .tools.qwen_tool import qwen_vision_tool        # Qwen Vision
# -------------------------------------------------------

from .tools.robo_control_node import servo_tool, move_robo  # Robot movement tools


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
#  FIXED TOOLS — PLAIN FUNCTIONS (ALL TOOLS MUST HAVE DOCSTRINGS)
# ==================================================================

_shared_agent_node = None


@tool
def speak_tool(message: str) -> str:
    """Publish a spoken message on the 'agent_response' topic and return confirmation."""
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
    """Perform a web search with Tavily and return a short summary of results."""
    log.info(f"[TOOL:tavily_tool] Query={query}")

    if tavily_client is None:
        return "Tavily not configured."

    try:
        res = tavily_client.search(query=query, max_results=max_results)
        results = res.get("results", [])
        if not results:
            return "No Tavily results."

        return "\n".join(
            f"- {r.get('title')} | {r.get('url')}\n  {r.get('content')[:200]}"
            for r in results
        )
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
    """Search the Qdrant vector database for relevant stored knowledge."""
    log.info(f"[TOOL:qdrant_search_tool] Query={query}")

    if q_vectorstore is None:
        log.error("[QDRANT] Vectorstore is None. Memory disabled.")
        return "Memory system not configured."

    try:
        matches = q_vectorstore.similarity_search(query, k=top_k)
        log.info(f"[QDRANT] Matches found: {len(matches)}")

        if not matches:
            return "No matching knowledge found."

        return "\n".join(
            f"- {doc.page_content} (source={doc.metadata.get('source','unknown')})"
            for doc in matches
        )
    except Exception as e:
        return f"Qdrant error: {str(e)}"






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

        # -------------------------------------------------------
        # ALL SUBAGENTS
        # -------------------------------------------------------
        self.subagents = [
            {
                "name": "research-subagent",
                "description": "Web research using Tavily.",
                "system_prompt": (
                    "Use tavily_tool for web research.\n"
                    "ALWAYS respond using speak_tool.\n"
                ),
                "tools": [tavily_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "memory-subagent",
                "description": "Personal Knowledge lookup from Qdrant.",
                "system_prompt": (
                    "Use qdrant_search_tool.\n"
                    "ALWAYS respond using speak_tool.\n"
                ),
                "tools": [qdrant_search_tool, speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "communication-subagent",
                "description": "Handles speaking.",
                "system_prompt": "Always use speak_tool to talk to user.",
                "tools": [speak_tool],
                "model": AGENT_MODEL,
            },
            {
                "name": "movement-subagent",
                "description": (
                    "Navigation + robot movement + servo control.\n"
                ),
                "system_prompt": (
                    "Use move_robo(F/B/L/R/S, value) for movement.\n"
                    "Use servo_tool(angle) for servo control.\n"
                    "Use vision subagent  if needed or YOLO if you know what you are looking.\n"
                    "Respond using speak_tool ALWAYS.\n"
                ),
                "tools": [describe_objects, move_robo, servo_tool, speak_tool],
                "model": AGENT_MODEL,
            },

            {
                "name": "vision-subagent",
                "description": "Handles YOLO, BLIP, and Qwen Vision tasks.",
                "system_prompt": (
                    "Use describe_objects() for object detection (YOLO).\n"
                    "Use describe_scene() for scene understanding (BLIP).\n"
                    "Use qwen_vision_tool(query) for advanced visual reasoning.\n"
                    "Always reply using speak_tool.\n"
                ),
                "tools": [
                    describe_objects,   # YOLO
                    describe_scene,     # BLIP
                    qwen_vision_tool,   # Qwen Vision
                    speak_tool
                ],
                "model": AGENT_MODEL,
            },
        ]

        # Master instructions
        self.system_prompt = SystemMessage(
            content=(
                "You are the robot's supervisor.\n"
                "Use speak_tool(message) for ALL responses.\n"
                "Delegate tasks to the best subagent.\n"
            )
        )

        # -------------------------------------------------------
        # REGISTER ALL TOOLS INCLUDING NEW VISION TOOLS
        # -------------------------------------------------------

        all_tools = [
            speak_tool,
            tavily_tool,
            qdrant_search_tool,
            describe_objects,
            describe_scene,
            qwen_vision_tool,
            servo_tool,
            move_robo,
        ]

        self.agent = create_deep_agent(
            model=AGENT_MODEL,
            tools=all_tools,
            subagents=self.subagents,
            system_prompt=self.system_prompt.content,
            backend=lambda rt: StateBackend(rt),
            store=self.store,
            checkpointer=self.checkpointer,
        )

        log.info("AgentNode initialized successfully.")

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

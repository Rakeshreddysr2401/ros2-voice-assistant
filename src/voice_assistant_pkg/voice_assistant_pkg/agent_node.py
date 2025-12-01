#!/usr/bin/env python3
"""
agent_node_main.py — Actual DeepAgents ROS2 Node

Subagents:
- research-subagent (Tavily)
- memory-subagent  (Qdrant)
- communication-subagent (speak_tool)

The agent decides when to speak by calling speak_tool(message).
This publishes to /agent_response → picked up by output_node (TTS).
"""

import os
import logging
from typing import Optional, Dict, Any, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# DeepAgents
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

# Tavily
from tavily import TavilyClient

# Qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("agent_node")


# ================================
# ENV VARS
# ================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_knowledge_base")
MODEL_NAME_EMBED = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")


# ================================
# SPEAK TOOL
# ================================
class SpeakWrapper:
    """
    Tool to publish text to agent_response topic.
    ROS2 publisher accessed through AgentNode._shared_instance
    """

    @tool
    def speak_tool(self, message: str) -> str:
        """
        Publish text to /agent_response so TTS can speak it.
        Returns confirmation to the LLM: "[spoken] text".
        """
        node = AgentNode._shared_instance
        if node is None:
            return "[speak_tool error: AgentNode not ready]"

        msg = String()
        msg.data = message
        node.pub_response.publish(msg)

        log.info(f"[Tool] speak_tool published: {message}")
        return f"[spoken] {message}"


speak_wrapper = SpeakWrapper()
speak_tool_fn = speak_wrapper.speak_tool


# ================================
# TAVILY TOOL
# ================================
class TavilyWrapper:

    def __init__(self):
        self.client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

    @tool
    def tavily_tool(self, query: str, max_results: int = 5, topic: str = "general") -> str:
        """
        Perform Tavily web research.
        """
        if not self.client:
            return "Tavily not configured."

        try:
            res = self.client.search(query=query, max_results=max_results, topic=topic)
            out = []
            for r in res.get("results", []):
                out.append(
                    f"- {r.get('title')} | {r.get('url')}\n  {r.get('content')[:200]}"
                )
            return "\n".join(out) if out else "No results."
        except Exception as e:
            return f"Tavily error: {str(e)}"


tavily_wrapper = TavilyWrapper()
tavily_tool_fn = tavily_wrapper.tavily_tool


# ================================
# QDRANT TOOL
# ================================
class QdrantWrapper:

    def __init__(self):
        if not (QDRANT_URL and QDRANT_API_KEY):
            self.vectorstore = None
            return

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embeddings = OpenAIEmbeddings(model=MODEL_NAME_EMBED)
        self.vectorstore = Qdrant(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embeddings=embeddings
        )

    @tool
    def qdrant_search_tool(self, query: str, top_k: int = QDRANT_TOP_K) -> str:
        """
        Retrieve personal knowledge using Qdrant similarity search.
        """
        if not self.vectorstore:
            return "Qdrant not configured."

        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            if not results:
                return f"No matches for '{query}'."

            out = []
            for doc in results:
                src = doc.metadata.get("source", "unknown")
                out.append(f"- {doc.page_content} (source: {src})")
            return "\n".join(out)
        except Exception as e:
            return f"Qdrant error: {str(e)}"


qdrant_wrapper = QdrantWrapper()
qdrant_tool_fn = qdrant_wrapper.qdrant_search_tool


# ================================
# AGENT NODE
# ================================
class AgentNode(Node):

    _shared_instance = None   # allows tools to publish to ROS topics

    def __init__(self):
        super().__init__("agent_node")
        AgentNode._shared_instance = self

        # ROS pubs/subs
        self.sub_input = self.create_subscription(String, "user_input", self._on_user_input, 10)
        self.pub_response = self.create_publisher(String, "agent_response", 10)

        # Memory + checkpointer
        self.store = InMemoryStore()
        self.checkpointer = MemorySaver()

        # Subagents
        self.subagents = [
            {
                "name": "research-subagent",
                "description": "Web research (Tavily).",
                "system_prompt": "Use tavily_tool to fetch information.",
                "tools": [tavily_tool_fn, speak_tool_fn],
                "model": AGENT_MODEL,
            },
            {
                "name": "memory-subagent",
                "description": "Personal knowledge retrieval via Qdrant.",
                "system_prompt": "Use qdrant_search_tool to retrieve Rakesh's knowledge.",
                "tools": [qdrant_tool_fn, speak_tool_fn],
                "model": AGENT_MODEL,
            },
            {
                "name": "communication-subagent",
                "description": "Handles speaking to the user.",
                "system_prompt": (
                    "For any user-facing message, call speak_tool(message). "
                    "Assistant text alone will not be spoken."
                ),
                "tools": [speak_tool_fn],
                "model": AGENT_MODEL,
            },
        ]

        # Supervisor prompt
        self.system_prompt = SystemMessage(
            content=(
                "You are the main humanoid robot brain.\n"
                "For any user-facing text, call speak_tool(message).\n"
                "Use research-subagent for web info.\n"
                "Use memory-subagent for Qdrant knowledge.\n"
                "Never assume plain assistant messages will be spoken — use speak_tool.\n"
            )
        )

        # Create agent
        self.agent = create_deep_agent(
            model=AGENT_MODEL,
            tools=[tavily_tool_fn, qdrant_tool_fn, speak_tool_fn],
            subagents=self.subagents,
            system_prompt=self.system_prompt.content,
            backend=lambda rt: StateBackend(rt),
            store=self.store,
            checkpointer=self.checkpointer,
        )

        self.get_logger().info("AgentNode with DeepAgents initialized.")

    # ============== handle user input ==============
    def _on_user_input(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        thread_id = "main_conversation"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = self.agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
            result = self._auto_resume_interrupts(result, config)

            # Fallback publishing if GPT emits assistant messages without speak_tool
            self._handle_fallback_messages(result)

        except Exception as e:
            err = String()
            err.data = f"Error: {str(e)}"
            self.pub_response.publish(err)

    # ============== auto-approve interrupts =========
    def _auto_resume_interrupts(self, result, config):
        while "__interrupt__" in result:
            interrupt = result["__interrupt__"][0].value
            decisions = [{"type": "approve"} for _ in interrupt["action_requests"]]
            result = self.agent.invoke(Command(resume={"decisions": decisions}), config=config)
        return result

    # ============== fallback assistant → speak =======
    def _handle_fallback_messages(self, result):
        msgs = result.get("messages", [])
        for m in msgs:
            if hasattr(m, "content"):
                text = (m.content or "").strip()
                if text and not text.startswith("[spoken]"):
                    # If the agent forgot to call speak_tool, we still speak it
                    msg = String()
                    msg.data = text
                    self.pub_response.publish(msg)
                    log.info(f"[Fallback Speak] {text}")


# ================================
# ROS2 MAIN
# ================================
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

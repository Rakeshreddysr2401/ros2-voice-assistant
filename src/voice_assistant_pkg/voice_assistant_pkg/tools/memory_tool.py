import os
import logging
from mem0 import Memory
from langchain_core.tools import tool

log = logging.getLogger("MemoryTool")

# Configuration for Mem0
# You can switch provider to "ollama" if you prefer local
LLM_PROVIDER = os.getenv("MEM0_LLM_PROVIDER", "openai") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://192.168.1.22:11434")

config = {
    "llm": {
        "provider": LLM_PROVIDER,
        "config": {
            "model": "gpt-4o-mini" if LLM_PROVIDER == "openai" else "qwen2.5:7b",
            "temperature": 0.1,
            "max_tokens": 1000,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": os.getenv("QDRANT_URL"),
            "api_key": os.getenv("QDRANT_API_KEY"),
            "collection_name": "mem0_knowledge_base", # Separate collection for structured facts
        }
    }
}

# Initialize Mem0
try:
    memory = Memory.from_config(config)
except Exception as e:
    log.error(f"Failed to initialize Mem0: {e}")
    memory = None

@tool
def add_memory_tool(fact: str) -> str:
    """
    Store a new fact or update an existing one in the robot's long-term memory.
    Use this when you learn something about the user, the environment, or object locations.
    Example: 'The red bottle is on the kitchen table'
    """
    if memory is None:
        return "Memory system not initialized."
    
    try:
        # user_id 'robot_boss' represents you (Rakesh)
        memory.add(fact, user_id="robot_boss")
        return f"✅ Remembered: {fact}"
    except Exception as e:
        return f"Error adding to memory: {str(e)}"

@tool
def search_memory_tool(query: str) -> str:
    """
    Search the robot's long-term memory for specific facts, preferences, or previous observations.
    Use this to recall where objects were last seen or user preferences.
    """
    if memory is None:
        return "Memory system not initialized."

    try:
        results = memory.search(query, user_id="robot_boss")
        if not results:
            return "No relevant memories found."
        
        # Format results nicely
        memory_strings = [f"- {m['memory']}" for m in results]
        return "Based on my memory:\n" + "\n".join(memory_strings)
    except Exception as e:
        return f"Error searching memory: {str(e)}"

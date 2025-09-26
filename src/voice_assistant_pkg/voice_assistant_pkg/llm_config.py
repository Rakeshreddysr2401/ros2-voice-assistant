# llm_config.py
from langchain_openai import ChatOpenAI
from .tools import get_tools
import os
from langchain_ollama import ChatOllama

MAC = "192.168.1.22:11434"  # your Mac Mini IP
def get_llm():
    """Initialize and return the LLM with proper configuration"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
        return None
    return ChatOllama(
        model="qwen2.5:7b",
        base_url=f"http://{MAC}"
    )

    # return ChatOpenAI(
    #     api_key=api_key,
    #     model="gpt-3.5-turbo",
    #     temperature=0.7,
    #     max_tokens=1000,
    #     timeout=30
    # )


def get_llm_with_tools():
    """Return LLM with tools bound to it"""
    llm = get_llm()
    if llm is None:
        return None

    tools = get_tools()
    return llm.bind_tools(tools)


# Initialize the LLM instances
llm = get_llm()
llm_with_tools = get_llm_with_tools()
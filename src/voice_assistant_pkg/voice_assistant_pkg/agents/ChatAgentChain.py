# chains/chatAgentChain.py
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from ..llm_config import llm_with_tools


def get_retry_prompt(retry_count: int, critique: str, suggestions: list[str] = None) -> SystemMessage:
    if suggestions is None:
        suggestions = []

    suggestion_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "No specific suggestions provided."

    return SystemMessage(content=f"""
RETRY ATTEMPT #{retry_count + 1}/2

The previous response had issues:
{critique.strip()}

Suggestions for improvement:
{suggestion_text}

Please revise your response carefully. Keep it concise (<120 words), relevant, and directly address the user's question.
""".strip())


# Chat chain using the LLM with tools
chat_chain = RunnableLambda(lambda messages: llm_with_tools.invoke(messages))
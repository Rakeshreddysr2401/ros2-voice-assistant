# agents/chatAgentNode.py
from langchain_core.messages import SystemMessage,AIMessage
from ..states.states import AgentState
from ..llm_config import llm_with_tools
from ..tools import build_system_message, get_tools


def call_agent(state: AgentState):

    try:
        messages = state["messages"]

        tools= get_tools()
        system_message = build_system_message(tools)

        print(f"Chat Agent called")
        # Ensure we have a system message at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            # Insert system message at the beginning
            conversation_messages = [system_message] + messages
        else:
            conversation_messages = messages

        # Get response from LLM
        response = llm_with_tools.invoke(conversation_messages)

        # Log tool usage for debugging
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tool_call.get('name', 'unknown') for tool_call in response.tool_calls]
            print(f"LLM requested tools: {tool_names}")

        return {
            "messages": messages + [response]
        }

    except Exception as e:
        print(f"Error in agent call: {str(e)}")
        error_response = AIMessage(
            content=f"I encountered an error while processing your request: {str(e)}. "
                    "Please try rephrasing your question or check if all services are running."
        )
        return {"messages": [error_response]}









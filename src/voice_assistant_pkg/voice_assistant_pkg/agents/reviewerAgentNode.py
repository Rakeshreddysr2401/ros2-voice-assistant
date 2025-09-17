# agents/reviewerAgentNode.py
from langgraph.config import get_stream_writer
from .reviewerAgentChain import ReviewFeedback, review_chain
from ..states.states import AgentState


MAX_RETRIES = 2

def reviewerAgent(state: AgentState):
    retry_count = state.get("retry_count", 0)
    messages = state["messages"]
    user_query = state.get("users_query")
    last_ai = state.get("chatAgentResponse")

    print(f"Reviewer Agent called : {retry_count+1} time")

    # If we've exceeded max retries, accept the current response and END
    if retry_count >= MAX_RETRIES:
        print(f"Max retries exceeded. Accepting final response: {last_ai}")
        writer = get_stream_writer()
        writer({"data": last_ai, "type": "final_response"})
        return {
            "final_response": last_ai,
            "review_feedback": {"satisfied": True, "reason": "max_retries_exceeded"}
        }

    # Build chat history string
    history_str = "\n".join(
        f"{msg.type.upper()}: {msg.content}"
        for msg in messages
        if hasattr(msg, 'content') and msg.content
    )

    try:
        feedback: ReviewFeedback = review_chain.invoke({
            "chat_history": history_str,
            "ai_response": last_ai.content if last_ai else "",
            "user_query": user_query or ""
        })
    except Exception as e:
        print(f"Error in review_chain.invoke: {e}")
        # If review fails, accept the response
        return {
            "final_response": last_ai,
            "review_feedback": {"satisfied": True, "reason": "review_error"}
        }

    feedback_dict = feedback.dict()
    satisfied = feedback_dict.get("satisfied", False)

    if satisfied:
        print("Satisfied by reviewer")
        writer = get_stream_writer()
        writer({"data": last_ai , "type": "final_response"})

    return {
        "final_response": last_ai if satisfied else state.get("final_response"),
        "review_feedback": feedback_dict,
        "retry_count": retry_count + 1 if not satisfied else retry_count,
    }

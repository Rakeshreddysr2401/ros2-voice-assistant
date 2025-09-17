from dotenv import load_dotenv
load_dotenv()

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from ..llm_config import llm


class ReviewFeedback(BaseModel):
    satisfied: Literal[True, False] = Field(..., description="False if the answer needs to be retried.")
    critique: Optional[str] = Field(default=None, description="Brief critique of the response, if any.")
    suggestions: Optional[List[str]] = Field(default=None, description="List of specific suggestions for improvement.")


parser = PydanticOutputParser(pydantic_object=ReviewFeedback)
format_instructions = parser.get_format_instructions()

review_prompt = PromptTemplate.from_template(
    """
You are a quality reviewer AI. You evaluate the assistant's response based on the full conversation history.

Checklist:
1. Does the response clearly and fully answer the user's query?
2. Is the response relevant and factually correct?
3. Is it under 120 words?
4. Should it have used tools to improve the answer?

If any issues exist, return `satisfied=False` with critique and suggestions.
If the response is good enough, return `satisfied=True`.

{format_instructions}

User's Original Query:
{user_query}

Conversation History:
{chat_history}

Final Assistant Response:
{ai_response}
""",
    partial_variables={"format_instructions": format_instructions}
)

# Final chain: Prompt → LLM → Pydantic parser
review_chain = review_prompt | llm | parser
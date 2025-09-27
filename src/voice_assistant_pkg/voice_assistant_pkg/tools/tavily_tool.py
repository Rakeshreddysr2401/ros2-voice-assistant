# #tools.tavily_tool.py
from langchain_core.tools import tool


from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilySearch(max_results=2)

@tool
def tavily_tool(query: str):
    """used it for general web search. like for unknown or realtime queries like current news, weather etc."""
    return tavily.invoke({"query": query})




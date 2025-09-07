from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def tavily_tool(query: str):
    """Search tasks, projects, or knowledge from the Tavily platform."""
    tavily_tool = TavilySearchResults(max_results=2)
    return tavily_tool.invoke({"query": query})







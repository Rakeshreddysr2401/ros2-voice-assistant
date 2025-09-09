from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Reuse the Tavily instance
tavily = TavilySearchResults(max_results=2)

@tool
def tavily_tool(query: str):
    """Search tasks, projects, or knowledge from the Tavily platform used as general web search."""
    return tavily.invoke({"query": query})

import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.tools import tool

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Vector store
vectorstore = Qdrant(
    client=client,
    collection_name="personal_knowledge_base",
    embeddings=embeddings
)

@tool
def qdrant_search_tool(query: str):
    """Search Rakesh's personal knowledge base for skills, projects, or experience."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "I couldn’t find anything in the knowledge base for that."
    return "\n\n".join([doc.page_content for doc in results])

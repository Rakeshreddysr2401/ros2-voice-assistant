import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_core.tools import tool

# Configurable params
K = int(os.getenv("QDRANT_TOP_K", 3))
MODEL_NAME = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Embeddings + Qdrant client
embeddings = OpenAIEmbeddings(model=MODEL_NAME)
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vectorstore = Qdrant(
    client=client,
    collection_name="personal_knowledge_base",
    embeddings=embeddings
)

@tool
def qdrant_search_tool(query: str):
    """Search Rakesh's personal knowledge base for skills, projects, or experience."""
    results = vectorstore.similarity_search(query, k=K)
    if not results:
        return "I couldn’t find anything in the knowledge base for that."
    return "\n\n".join([
        f"- {doc.page_content} (source: {doc.metadata.get('source', 'unknown')})"
        for doc in results
    ])

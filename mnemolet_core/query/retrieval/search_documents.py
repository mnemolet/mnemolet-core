from .qdrant_retriever import QdrantRetriever


def search_documents(query: str, top_k: int):
    """
    Wrapper around QdrantRetriever.
    """
    xz = QdrantRetriever()
    results = xz.search(query, top_k)
    return results

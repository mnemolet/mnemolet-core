from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class QdrantRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", url="http://localhost:6333"):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(url)

    def search(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name="documents", query_vector=query_vector, limit=top_k
        )
        return [
            {
                "text": i.payload["text"],
                "score": i.score,
            }
            for i in results
        ]

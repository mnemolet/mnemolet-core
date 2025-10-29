from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class QdrantRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", url="http://localhost:6333"):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(url)

    def search(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(query).tolist()

        results = self.client.query_points(
            collection_name="documents",
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": i.payload.get("text", ""),
                "score": i.score,
                "path": i.payload.get("path", ""),
                "hash": i.payload.get("hash", ""),
            }
            for i in results.points
        ]

from typing import Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from mnemolet_core.config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL


class QdrantRetriever:
    def __init__(self, model_name=EMBED_MODEL, url=QDRANT_URL):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(url)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
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

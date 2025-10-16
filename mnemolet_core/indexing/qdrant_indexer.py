from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams


class QdrantIndexer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents",
    ):
        self.client = QdrantClient(url=f"http//{host}:{port}")
        self.collection_name = collection_name

    def init_collection(self, vector_size: int = 384):
        """
        Create Qdrant collection.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def store_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """
        Store text embeddings in Qdrant.
        """
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={"text": texts[i]},
            )
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

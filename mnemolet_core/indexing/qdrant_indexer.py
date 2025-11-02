import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from mnemolet_core.config import QDRANT_URL, QDRANT_COLLECTION


class QdrantIndexer:
    def __init__(self):
        """
        Init Qdrant client using config.toml.
        """
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = QDRANT_COLLECTION

    def init_collection(self, vector_size: int = 384):
        """
        Create Qdrant collection.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def store_embeddings(
        self, chunks: list[str], embeddings: np.ndarray, metadata: list[dict[str, str]]
    ):
        """
        Store text embeddings in Qdrant.
        """
        payloads = [
            {"path": m["path"], "hash": m["hash"], "text": chunk}
            for m, chunk in zip(metadata, chunks)
        ]

        # build Qdrand points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload=payloads[i],
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

        info = self.client.get_collection(self.collection_name)
        points_count = info.points_count
        indexed_count = info.indexed_vectors_count
        print(f"Upserted → total points: {points_count}, indexed: {indexed_count}")

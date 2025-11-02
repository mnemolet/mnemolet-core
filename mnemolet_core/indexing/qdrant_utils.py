from mnemolet_core.config import QDRANT_URL, QDRANT_COLLECTION
from qdrant_client import QdrantClient


def get_collection_stats(collection_name: str = QDRANT_COLLECTION) -> dict:
    """
    Return collection stats as a dictionary.
    """
    client = QdrantClient(url=QDRANT_URL)
    info = client.get_collection(collection_name)

    return {
        "collection_name": collection_name,
        "status": info.status,
        "optimizer_status": info.optimizer_status,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "segment_count": info.segments_count,
        "vector_size": info.config.params.vectors.size,
        "distance": info.config.params.vectors.distance,
        "on_disk_payload": info.config.params.on_disk_payload,
    }


def remove_collection(collection_name: str = QDRANT_COLLECTION) -> None:
    client = QdrantClient(url=QDRANT_URL)
    client.delete_collection(collection_name=collection_name)

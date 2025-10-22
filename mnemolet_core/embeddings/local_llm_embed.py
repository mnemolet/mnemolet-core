from typing import List
from sentence_transformers import SentenceTransformer
import torch

# detect GPU automatically, small embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using small embedding model.
    Current implementation is not efficient, it is done only for initial MVP.
    It loads all texts into memory and it won't work with large dataset.
    TODO: refactor using batch processing

    Args:
        texts: List of text chunks to embed

    Returns:
        List of embeddings
    """
    if not texts:
        return []

    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()

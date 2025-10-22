from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# detect GPU automatically, small embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using small embedding model.
    Current implementation is not efficient, it is done only for initial MVP.
    It loads all texts into memory and it won't work with large dataset.
    TODO: refactor using batch processing

    Args:
        texts: List of text chunks to embed
        batch_size: Number of items per batch

    Returns:
        List of embeddings
    """
    if not texts:
        return []

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device
        )
        embeddings.append(batch_embeddings)
    all_embeddings = np.vstack(embeddings)

    return all_embeddings.tolist()

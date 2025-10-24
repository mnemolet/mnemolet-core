from typing import Iterable, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os

# detect GPU automatically, small embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def embed_texts(
    texts: Iterable[str],
    output_file: str,
    batch_size: int = 512,
) -> Tuple[int, int]:
    """
    Generate embeddings in batches and save directly to disk.

    Args:
        texts: Iterable of text chunks to embed
        output_file: Path to save the embeddings (*.npy)
        batch_size: Number of items per batch

    Returns:
        a tuple (num_texts, embedding_dim) with the total number of chunks and
        the embedding vector dimension.
    """
    # remove existing embeddings
    if os.path.exists(output_file):
        os.remove(output_file)

    texts = list(texts)
    embedding_dim = model.get_sentence_embedding_dimension()
    num_texts = len(texts)

    # preallocate memmap array
    embeddings_memmap = np.memmap(
        output_file, dtype=np.float32, mode="w+", shape=(num_texts, embedding_dim)
    )

    for i in tqdm(range(0, num_texts, batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(
            batch, convert_to_numpy=True, show_progress_bar=False, device=device
        )
        batch_embeddings = np.stack(batch_embeddings).astype(np.float32)
        print(batch_embeddings.shape)
        embeddings_memmap[i : i + len(batch), :] = batch_embeddings

    embeddings_memmap.flush()
    return num_texts, embedding_dim

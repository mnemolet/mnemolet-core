from pathlib import Path

from mnemolet_core.ingestion.txt_loader import load_txt_files
from mnemolet_core.ingestion.preprocessor import chunk_text
from mnemolet_core.embeddings.local_llm_embed import embed_texts
from mnemolet_core.indexing.qdrant_indexer import QdrantIndexer
from mnemolet_core.query.retriever import QdrantRetriever
from mnemolet_core.query.generator import LocalGenerator
from mnemolet_core.storage import db_tracker
import numpy as np
import json


def ingest(directory: str, force: bool = False):
    """Ingest .txt files from a directory into Qdrant."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: directory {directory} not found")
        return

    db_tracker.init_db()

    texts_data = load_txt_files(directory)
    new_texts = []

    for data in texts_data:
        if force or not db_tracker.file_exists(data["hash"]):
            db_tracker.add_file(data["path"], data["hash"])
            new_texts.append(data)

    if not new_texts:
        print("No new files to ingest.")
        return

    print(f"Processing {len(new_texts)} new files...")

    total_chunks = 0
    all_chunks = []
    for data in new_texts:
        chunks = chunk_text(data["content"])
        total_chunks += len(chunks)
        all_chunks.extend(chunks)

        # DEBUG: print info about chunks
        print(f"{data['path']}: {len(chunks)} chunks")
        for i, c in enumerate(chunks[:2], 1):  # show first 2
            print(f"  Chunk {i}: {c[:80]}...")

    print(f"Total chunks: {total_chunks}")

    # generate embeddings for all chunks and save to file
    embed_texts(all_chunks, output_file="embeddings.npy")
    # memory-map embeddings
    with open("embeddings_metadata.json", "r") as f:
        metadata = json.load(f)
    embeddings = np.memmap(
        "embeddings.npy",
        dtype=np.float32,
        mode="r",
        shape=(metadata["num_texts"], metadata["embedding_dim"]),
    )

    print(f"Generated embeddings for {len(embeddings)} chunks.")
    print(f"Example embedding (first chunk, first 8 values): {embeddings[0][:8]}")

    # store in Qdrant
    indexer = QdrantIndexer()
    if force:
        print("Recreating Qdrant collection..")
        indexer.init_collection(vector_size=len(embeddings[0]))

    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        chunk_batch = all_chunks[i : i + batch_size]
        emb_batch = embeddings[i : i + batch_size]
        indexer.store_embeddings(chunk_batch, emb_batch)
    print("Stored embeddings in Qdrant successfully.")


def search(q: str, top_k: int = 5):
    retriever = QdrantRetriever()
    results = retriever.search(q, top_k=top_k)

    if not results:
        print("No results found.")
        return

    print("\nTop results:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. (score={r['score']:.4f}) {r['text'][:200]}...\n")


def answer(q: str, top_k: int = 3):
    """
    Search Qdrant and generate an answer using local LLM.
    """
    retriever = QdrantRetriever()
    generator = LocalGenerator("llama3")

    print(f"Searching for top {top_k} results..")
    results = retriever.search(q, top_k=top_k)

    if not results:
        print("No results found.")
        return

    context_chunks = [r["text"] for r in results]
    print("Generating answer..")
    answer_text = generator.generate_answer(q, context_chunks)
    print("\nAnswer:\n")
    print(answer_text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print(" python -m cli.main ingest <directory> [--force]")
        print(" python -m cli.main search <query> [--top-k N]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python -m cli.main ingest <directory> [--force]")
            sys.exit(1)
        data_dir = sys.argv[2]
        force_flag = "--force" in sys.argv
        ingest(data_dir, force=force_flag)

    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python -m cli.main search <query> [--top-k N]")
            sys.exit(1)

        q = sys.argv[2]
        top_k = 5
        if "--top-k" in sys.argv:
            idx = sys.argv.index("--top-k")
            if idx + 1 < len(sys.argv):
                top_k = int(sys.argv[idx + 1])
        search(q, top_k=top_k)

    elif command == "answer":
        if len(sys.argv) < 3:
            print("Usage: python -m cli.main answer <query>")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        answer(query)

    else:
        print(f"Unknown command: {command}")

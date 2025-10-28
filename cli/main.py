import click
import time
from pathlib import Path

from mnemolet_core.ingestion.preprocessor import process_directory
from mnemolet_core.embeddings.local_llm_embed import embed_texts_batch
from mnemolet_core.indexing.qdrant_indexer import QdrantIndexer
from mnemolet_core.query.retriever import QdrantRetriever
from mnemolet_core.query.generator import LocalGenerator
from mnemolet_core.storage import db_tracker


def ingest(directory: str, force: bool = False, batch_size: int = 100):
    """
    Ingest files from a directory into Qdrant.
    - streams files and chunks
    """
    start_total = time.time()
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: directory {directory} not found")
        return

    print(f"Starting ingestion from {directory}")
    db_tracker.init_db()
    indexer = QdrantIndexer()
    embedding_dim = None
    first_batch = True
    total_chunks = 0
    total_files = 0

    chunk_batch = []
    metadata_batch = []

    for data in process_directory(directory):
        file_path = data["path"]
        file_hash = data["hash"]
        chunk = data["chunk"]
        # skip files already processed
        if not force and db_tracker.file_exists(file_hash):
            print(f"Skipping already ingested: {file_path}")
            continue

        is_new_file = (
            not metadata_batch  # first chunk overall
            or file_path != metadata_batch[-1]["path"]  # file changed
        )

        if is_new_file:
            print(f"Processing file #{total_files}: {file_path}")
            total_files += 1

        db_tracker.add_file(data["path"], data["hash"])

        # add to current batch
        chunk_batch.append(chunk)
        metadata_batch.append({"path": file_path, "hash": file_hash})
        total_chunks += 1

        # if batch full —> embed & store
        if len(chunk_batch) >= batch_size:
            print(f"Embedding batch of {len(chunk_batch)} chunks..")
            for embeddings in embed_texts_batch(chunk_batch, batch_size=batch_size):
                if first_batch:
                    embedding_dim = embeddings.shape[1]
                    if force:
                        print(f"Recreating Qdrant collection (dim={embedding_dim})..")
                        indexer.init_collection(vector_size=embedding_dim)
                    first_batch = False

                indexer.store_embeddings(chunk_batch, embeddings, metadata_batch)
                print(f"Stored {len(chunk_batch)} chunks in Qdrant.")
                chunk_batch.clear()
                metadata_batch.clear()

    # handle the rest
    if chunk_batch:
        print(f"Final batch: embedding {len(chunk_batch)} chunks..")
        for embeddings in embed_texts_batch(chunk_batch, batch_size=batch_size):
            if first_batch:
                embedding_dim = embeddings.shape[1]
                if force:
                    print("Recreating Qdrant collection (dim={embedding_dim})..")
                    indexer.init_collection(vector_size=embedding_dim)
                first_batch = False

            indexer.store_embeddings(chunk_batch, embeddings, metadata_batch)
            print(f"Stored {len(chunk_batch)} chunks in Qdrant.")
            chunk_batch.clear()
            metadata_batch.clear()

    total_time = time.time() - start_total

    print(
        f"Ingestion complete: {total_files} files, {total_chunks} stored in "
        f"Qdrant in {total_time:.1f}s.\n"
    )


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
    # for DEBUG only
    print("Raw result sample:\n", results[0])

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

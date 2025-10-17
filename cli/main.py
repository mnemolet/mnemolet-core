from pathlib import Path

from mnemolet_core.ingestion.txt_loader import load_txt_files
from mnemolet_core.ingestion.preprocessor import chunk_text
from mnemolet_core.embeddings.local_llm_embed import embed_texts
from mnemolet_core.indexing.qdrant_indexer import QdrantIndexer
from mnemolet_core.storage import db_tracker


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

    # generate embeddings for all chunks
    embeddings = embed_texts(all_chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks.")
    print(f"Example embedding (first chunk, first 8 values): {embeddings[0][:8]}")

    # store in Qdrant
    indexer = QdrantIndexer()
    indexer.init_collection(vector_size=len(embeddings[0]))
    indexer.store_embeddings(all_chunks, embeddings)
    print("Stored embeddings in Qdrant successfully.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m cli.main <directory> [--force]")
        sys.exit(1)

    data_dir = sys.argv[1]
    force_flag = "--force" in sys.argv
    ingest(data_dir, force=force_flag)

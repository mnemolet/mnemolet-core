import click
import time
from pathlib import Path

from mnemolet_core.ingestion.preprocessor import process_directory
from mnemolet_core.embeddings.local_llm_embed import embed_texts_batch
from mnemolet_core.indexing.qdrant_indexer import QdrantIndexer
from mnemolet_core.indexing.qdrant_utils import get_collection_stats, remove_collection
from mnemolet_core.query.retriever import QdrantRetriever
from mnemolet_core.query.generator import LocalGenerator
from mnemolet_core.storage import db_tracker


@click.group()
def cli():
    """
    CLI for mnemolet.
    """
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--force", is_flag=True, help="Recreate Qdrant collection and reindex all files."
)
@click.option(
    "--batch-size", default=100, show_default=True, help="Number of chunks per batch."
)
def ingest(directory: str, force: bool, batch_size: int):
    """
    Ingest files from a directory into Qdrant.
    - streams files, chunks them, embeds text and stores data in Qdrant.
    """
    start_total = time.time()
    directory = Path(directory)

    click.echo(f"Starting ingestion from {directory}")
    db_tracker.init_db()
    indexer = QdrantIndexer()
    embedding_dim = None
    # first_batch = True
    total_chunks = 0
    total_files = 0

    chunk_batch = []
    metadata_batch = []

    seen_files = set()

    if force:
        first_chunk = next(process_directory(directory))
        first_embedding = next(embed_texts_batch([first_chunk["chunk"]], batch_size=1))
        embedding_dim = first_embedding.shape[1]
        click.echo(f"Recreating Qdrant collection (dim={embedding_dim})..")
        indexer.init_collection(vector_size=embedding_dim)

    for data in process_directory(directory):
        file_path = data["path"]
        file_hash = data["hash"]
        chunk = data["chunk"]

        # skip files already processed
        if not force and db_tracker.file_exists(file_hash):
            click.echo(f"Skipping already ingested: {file_path}")
            continue

        if file_path not in seen_files:
            click.echo(f"Processing file #{total_files}: {file_path}")
            total_files += 1
            seen_files.add(file_path)

        db_tracker.add_file(data["path"], data["hash"])

        # add to current batch
        chunk_batch.append(chunk)
        metadata_batch.append({"path": file_path, "hash": file_hash})
        total_chunks += 1

        # if batch full —> embed & store
        if len(chunk_batch) >= batch_size:
            _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force)
            # first_batch = False
            chunk_batch.clear()
            metadata_batch.clear()

    # handle the rest
    if chunk_batch:
        _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force)

    total_time = time.time() - start_total

    click.echo(
        f"Ingestion complete: {total_files} files, {total_chunks} stored in "
        f"Qdrant in {total_time:.1f}s.\n"
    )


def _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force):
    click.echo(f"Embedding batch of {len(chunk_batch)} chunks..")
    for embeddings in embed_texts_batch(chunk_batch, batch_size=len(chunk_batch)):
        indexer.store_embeddings(chunk_batch, embeddings, metadata_batch)
        click.echo(f"Stored {len(chunk_batch)} chunks in Qdrant.")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--top-k", default=5, show_default=True, help="Number of results to retrieve."
)
def search(query: str, top_k: int):
    """
    Search Qdrant for relevant documents.
    """
    retriever = QdrantRetriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        click.echo("No results found.")
        return

    click.echo("\nTop results:\n")
    for i, r in enumerate(results, start=1):
        click.echo(
            f"{i}. (score={r['score']:.4f}) (path={r['path']}) {r['text'][:200]}...\n"
        )


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--top-k",
    default=3,
    show_default=True,
    help="Number of context chunks for generation.",
)
@click.option(
    "--model",
    default="llama3",
    show_default=True,
    help="Local model to use for generation.",
)
def answer(query: str, top_k: int, model: str):
    """
    Search Qdrant and generate an answer using local LLM.
    """
    retriever = QdrantRetriever()
    generator = LocalGenerator(model)

    click.echo(f"Searching for top {top_k} results..")
    results = retriever.search(query, top_k=top_k)

    # for DEBUG only
    # print("Raw result sample:\n", results[0])

    if not results:
        click.echo("No results found.")
        return

    context_chunks = [r["text"] for r in results]
    click.echo("Generating answer..")

    answer_text = generator.generate_answer(query, context_chunks)
    click.echo("\nAnswer:\n")
    click.echo(answer_text)

    click.echo("\nSources:\n")
    results = _only_unique(results)
    for i, r in enumerate(results, start=1):
        click.echo(f"{i}. {r['path']} (score={r['score']:.4f})")


def _only_unique(xz: list) -> list:
    """
    Helper fn to return only unique results by file path.
    """
    unique = []
    seen = set()

    for x in xz:
        path = x["path"]
        if path not in seen:
            seen.add(path)
            unique.append(x)
    return unique


@cli.command()
@click.option(
    "--collection_name",
    default="documents",
    help="Define collection name.",
)
def stats(collection_name: str):
    """
    Output statistics about Qdrant database.
    """
    try:
        stats = get_collection_stats(collection_name)
    except Exception as e:
        click.echo(f"Failed to fetch stats: {e}")
        return

    click.echo(f"Qdrant Collection Stats: {stats['collection_name']}")
    click.echo("-" * 60)
    for k, v in stats.items():
        if k != "collection_name":
            click.echo(f"{k.replace('_', ' ').title():22}: {v}")
    click.echo("-" * 60)


@cli.command()
@click.option(
    "--collection_name",
    default="documents",
    help="Define collection name.",
)
def remove(collection_name: str):
    """
    Remove Qdrant collection.
    """
    click.confirm(
        f"Are you sure you want to delete the collection '{collection_name}'?",
        abort=True,
    )
    try:
        remove_collection(collection_name)
        click.echo(f"Collection '{collection_name}' removed successfully.")
    except Exception as e:
        click.echo(f"Failed to remove collection '{collection_name}': {e}")


if __name__ == "__main__":
    # import gc
    # import warnings

    # warnings.filterwarnings("always", category=ResourceWarning)
    # gc.collect()
    cli()

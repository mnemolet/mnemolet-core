import logging
import sys
import time
from functools import wraps
from pathlib import Path

import click
from tqdm import tqdm

from mnemolet_core.config import (
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)
from mnemolet_core.embeddings.local_llm_embed import embed_texts_batch
from mnemolet_core.indexing.qdrant_indexer import QdrantIndexer
from mnemolet_core.indexing.qdrant_utils import (
    get_collection_stats,
    list_collections,
    remove_collection,
)
from mnemolet_core.ingestion.preprocessor import process_directory
from mnemolet_core.query.generation.generate_answer import generate_answer
from mnemolet_core.query.retrieval.search_documents import search_documents
from mnemolet_core.storage import db_tracker
from mnemolet_core.utils.qdrant import check_qdrant_status

logger = logging.getLogger(__name__)


def requires_qdrant(f):
    """
    Decorator to check Qdrant before running a command.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not check_qdrant_status(QDRANT_URL):
            sys.exit(1)
        return f(*args, **kwargs)

    return wrapper


@click.group()
@click.option(
    "-v", "--verbose", count=True, help="Increase output verbosity (use -v or -vv)"
)
@click.pass_context
def cli(ctx, verbose):
    """
    CLI for mnemolet.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # map -v to INFO, --vv to DEBUG
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:  # -vv or more
        level = logging.DEBUG

    # reset existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    logger.debug(f"Logger init with level={level}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--force", is_flag=True, help="Recreate Qdrant collection and reindex all files."
)
@click.option(
    "--batch-size", default=100, show_default=True, help="Number of chunks per batch."
)
@click.pass_context
@requires_qdrant
def ingest(ctx, directory: str, force: bool, batch_size: int):
    """
    Ingest files from a directory into Qdrant.
    - streams files, chunks them, embeds text and stores data in Qdrant.
    """
    start_total = time.time()
    directory = Path(directory)

    files = list(directory.rglob("*"))
    files = [f for f in files if f.is_file()]
    if not files:
        logger.warning("No files found to ingest.")
        return
    logger.info(f"Found {len(files)} files to ingest from {directory}.")

    logger.info(f"Starting ingestion from {directory}")
    db_tracker.init_db()
    indexer = QdrantIndexer(QDRANT_URL, QDRANT_COLLECTION)
    embedding_dim = None
    total_chunks = 0
    total_files = 0  # can be actually different with files count

    chunk_batch = []
    metadata_batch = []

    pbar = tqdm(total=len(files), desc="Ingesting files", unit="file")

    seen_files = set()

    if force:
        first_chunk = next(process_directory(directory))
        first_embedding = next(embed_texts_batch([first_chunk["chunk"]], batch_size=1))
        embedding_dim = first_embedding.shape[1]
        logger.info(f"Recreating Qdrant collection (dim={embedding_dim})..")
        indexer.init_collection(vector_size=embedding_dim)

    for data in process_directory(directory):
        file_path = data["path"]
        file_hash = data["hash"]
        chunk = data["chunk"]

        # skip files already processed
        if not force and db_tracker.file_exists(file_hash):
            logger.info(f"Skipping already ingested: {file_path}")
            continue

        if file_path not in seen_files:
            logger.info(f"Processing file #{total_files}: {file_path}")
            total_files += 1
            seen_files.add(file_path)
            pbar.update(1)  # increment progress bar

        db_tracker.add_file(data["path"], data["hash"])

        # add to current batch
        chunk_batch.append(chunk)
        metadata_batch.append({"path": file_path, "hash": file_hash})
        total_chunks += 1

        # if batch full —> embed & store
        if len(chunk_batch) >= batch_size:
            _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force)
            chunk_batch.clear()
            metadata_batch.clear()

    # handle the rest
    if chunk_batch:
        _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force)

    pbar.close()

    total_time = time.time() - start_total

    click.echo(
        f"Ingestion complete: {total_files} files, {total_chunks} stored in "
        f"Qdrant in {total_time:.1f}s.\n"
    )


def _store_batch(indexer, chunk_batch, metadata_batch, embedding_dim, force):
    logger.info(f"Embedding batch of {len(chunk_batch)} chunks..")
    for embeddings in embed_texts_batch(chunk_batch, batch_size=len(chunk_batch)):
        indexer.store_embeddings(chunk_batch, embeddings, metadata_batch)
        logger.info(f"Stored {len(chunk_batch)} chunks in Qdrant.")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--top-k", default=TOP_K, show_default=True, help="Number of results to retrieve."
)
@requires_qdrant
def search(query: str, top_k: int):
    """
    Search Qdrant for relevant documents.
    """
    results = search_documents(query, top_k=top_k)

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
    default=TOP_K,
    show_default=True,
    help="Number of context chunks for generation.",
)
@click.option(
    "--model",
    default="llama3",
    show_default=True,
    help="Local model to use for generation.",
)
@requires_qdrant
def answer(query: str, top_k: int, model: str):
    """
    Search Qdrant and generate an answer using local LLM.
    """
    click.echo("Generating answer..")
    results = search_documents(query, top_k)
    answer = generate_answer(query, top_k)

    click.echo("\nAnswer:\n")
    click.echo(answer)

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
    default=QDRANT_COLLECTION,
    help="Define collection name.",
)
@requires_qdrant
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
    default=QDRANT_COLLECTION,
    help="Define collection name.",
)
@requires_qdrant
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


@cli.command()
@requires_qdrant
def list_collections_cli():
    """
    List all Qdrant collections.
    """
    xz = list_collections()
    if not xz:
        click.echo("No collections found.")
    else:
        click.echo("Collections in Qdrant:")
        for x in xz:
            click.echo(f"- {x}")


if __name__ == "__main__":
    cli()

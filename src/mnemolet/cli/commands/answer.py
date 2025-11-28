import logging

import click

from mnemolet.config import (
    EMBED_MODEL,
    MIN_SCORE,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    TOP_K,
)

from .utils import requires_qdrant

logger = logging.getLogger(__name__)


@click.command()
@click.argument("query", type=str)
@click.option(
    "--top-k",
    default=TOP_K,
    show_default=True,
    help="Number of context chunks for generation.",
)
@click.option(
    "--ollama-url",
    default=OLLAMA_URL,
    show_default=True,
    help="Ollama url",
)
@click.option(
    "--ollama-model",
    default="llama3",
    show_default=True,
    help="Local model to use for generation.",
)
@click.option(
    "--min-score", default=MIN_SCORE, show_default=True, help="Minimum score threshold."
)
@requires_qdrant
def answer(
    ollama_url: str, query: str, top_k: int, ollama_model: str, min_score: float
):
    """
    Search Qdrant and generate an answer using local LLM.
    """
    from mnemolet.core.query.generation.generate_answer import generate_answer

    click.echo("Generating answer..")

    final_results = []

    for chunk, results in generate_answer(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        embed_model=EMBED_MODEL,
        ollama_url=OLLAMA_URL,
        model=ollama_model,
        query=query,
        top_k=top_k,
        min_score=min_score,
    ):
        if results is None:
            click.echo(chunk, nl=False)
        else:
            final_results = results

    click.echo("\n")

    if final_results:
        click.echo("\nSources:\n")
        for i, r in enumerate(final_results, start=1):
            click.echo(f"{i}. {r['path']} (score={r['score']:.4f})")

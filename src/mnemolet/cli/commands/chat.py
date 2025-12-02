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
def chat(ollama_url: str, top_k: int, ollama_model: str, min_score: float):
    """
    Start interactive chat session with the local LLM.
    """
    from mnemolet.core.query.generation.chat_session import ChatSession
    from mnemolet.core.query.retrieval.retriever import Retriever, RetrieverConfig

    retriever_cfg = RetrieverConfig(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        embed_model=EMBED_MODEL,
        top_k=top_k,
        min_score=min_score,
    )
    retriever = Retriever(retriever_cfg)

    session = ChatSession(
        retriever=retriever,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )

    click.echo("Starting chat. Type 'exit' to quit.\n")

    while True:
        query = click.prompt("> ", type=str)

        if query.lower() in ("exit", "quit", ":q"):
            click.echo("Bye")
            break

        # stream response
        click.echo("assistant: ", nl=False)

        for chunk in session.ask(query):
            click.echo(chunk, nl=False)

    click.echo()

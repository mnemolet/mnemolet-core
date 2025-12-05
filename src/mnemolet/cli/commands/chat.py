import logging

import click

from mnemolet.cli.commands.chat_history import history
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


@click.group()
def chat():
    """
    Chat commands.
    """
    pass


chat.add_command(history)


@chat.command("start")
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
def start(ollama_url: str, top_k: int, ollama_model: str, min_score: float):
    """
    Start interactive chat session with the local LLM.
    """
    from mnemolet.core.query.generation.chat_session import ChatSession
    from mnemolet.core.query.generation.local_generator import get_llm_generator
    from mnemolet.core.query.retrieval.retriever import get_retriever

    retriever = get_retriever(
        url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        model=EMBED_MODEL,
        top_k=top_k,
        min_score=min_score,
    )

    generator = get_llm_generator(OLLAMA_URL, ollama_model)

    session = ChatSession(
        retriever=retriever,
        generator=generator,
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

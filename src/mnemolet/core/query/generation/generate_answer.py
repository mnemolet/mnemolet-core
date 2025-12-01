import logging
from typing import Generator, Optional, Tuple

from mnemolet.core.query.generation.local_generator import LocalGenerator
from mnemolet.core.query.retrieval.search_documents import search_documents
from mnemolet.core.utils.utils import filter_by_min_score

logger = logging.getLogger(__name__)


def generate_answer(
    qdrant_url: str,
    collection_name: str,
    embed_model: str,
    ollama_url: str,
    model: str,
    query: str,
    top_k: int,
    min_score: float,
    chat: bool = False,  # default for answer endpoint
) -> Generator[Tuple[str, Optional[list[dict]]], None, None]:
    """
    Wrapper around LocalGenerator.
    """
    logger.info(f"Searching for top {top_k} results..")
    # ------- Retrieval -------
    results = search_documents(
        qdrant_url, collection_name, embed_model, query, top_k=top_k
    )

    filtered_results = filter_by_min_score(results, min_score)

    # ------- Answer mode -------
    if not chat:
        if not filtered_results:
            yield "No relevant information found.", []
            return

        generator = LocalGenerator(ollama_url, model)
        context_chunks = [r["text"] for r in filtered_results]
        logger.info("Generating answer..")

        # strem all chunks directly
        yield from (
            (chunk, None) for chunk in generator.generate_answer(query, context_chunks)
        )

        # finally send sources
        yield "", _only_unique(filtered_results)
        return

    # ------- Chat mode -------
    if filtered_results:
        logger.info("Relevant context found for chat.")
        context_chunks = [r["text"] for r in filtered_results]
    else:
        logger.info("No relevant context found; continue chat without context.")
        context_chunks = []

    generator = LocalGenerator(ollama_url, model)

    logger.info("Generating chat response...")

    # stream LLM output
    for c in generator.generate_answer(query, context_chunks):
        yield c, None

    # return sources only if we had any
    if filtered_results:
        yield "", _only_unique(filtered_results)
    else:
        yield "", []


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

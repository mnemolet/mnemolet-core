import logging
from typing import Generator, Optional, Tuple

from mnemolet.core.query.generation.local_generator import LocalGenerator
from mnemolet.core.query.retrieval.search_documents import search_documents
from mnemolet.core.utils.utils import _only_unique, filter_by_min_score

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
    filtered_results = _retrieve_context(
        qdrant_url, collection_name, embed_model, query, top_k, min_score
    )

    # ------- Answer mode -------
    if not chat:
        if not filtered_results:
            yield "No relevant information found.", []
            return

        # generator = LocalGenerator(ollama_url, model)
        context_chunks = [r["text"] for r in filtered_results]
        logger.info("Generating answer..")

        # stream LLM output
        for c in _generate_llm_chunks(ollama_url, model, query, context_chunks):
            yield c, None

        # finally send sources
        yield _yield_sources_if_any(filtered_results)
        return

    # ------- Chat mode -------
    if filtered_results:
        logger.info("Relevant context found for chat.")
        context_chunks = [r["text"] for r in filtered_results]
    else:
        logger.info("No relevant context found; continue chat without context.")
        context_chunks = []

    logger.info("Generating chat response...")

    # stream LLM output
    for c in _generate_llm_chunks(ollama_url, model, query, context_chunks):
        yield c, None

    # return sources only if we had any
    yield _yield_sources_if_any(filtered_results)


def _retrieve_context(
    qdrant_url: str,
    collection_name: str,
    embed_model: str,
    query: str,
    top_k: int,
    min_score: float,
) -> list[dict]:
    """
    Helper fn for retrieving and filter context chunks from Qdrant.
    """
    results = search_documents(
        qdrant_url, collection_name, embed_model, query, top_k=top_k
    )
    return filter_by_min_score(results, min_score)


def _generate_llm_chunks(
    ollama_url: str, model: str, query: str, context_chunks: list[str]
) -> Generator[str, None, None]:
    """
    Helper fn for streaming chunks from LocalGenerator.
    """
    generator = LocalGenerator(ollama_url, model)
    yield from generator.generate_answer(query, context_chunks)


def _yield_sources_if_any(
    filtered_results: list[dict],
) -> Tuple[str, Optional[list[dict]]]:
    """
    Return sources if they exist, else empty list.
    """
    if filtered_results:
        return "", _only_unique(filtered_results)
    return "", []

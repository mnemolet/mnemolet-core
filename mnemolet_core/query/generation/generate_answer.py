import logging

from ..retrieval.search_documents import search_documents
from .local_generator import LocalGenerator

logger = logging.getLogger(__name__)


def generate_answer(
    qdrant_url: str,
    collection_name: str,
    embed_model: str,
    ollama_url: str,
    model: str,
    query: str,
    top_k: int,
) -> tuple[str, list[dict]]:
    """
    Wrapper around LocalGenerator.
    """
    logger.info(f"Searching for top {top_k} results..")
    results = search_documents(
        qdrant_url, collection_name, embed_model, query, top_k=top_k
    )

    if not results:
        return "No relevant information found.", []

    generator = LocalGenerator(ollama_url, model)
    context_chunks = [r["text"] for r in results]
    logger.info("Generating answer..")

    answer = generator.generate_answer(query, context_chunks)
    results = _only_unique(results)

    return answer, results


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

import logging

from ..retrieval.search_documents import search_documents
from .local_generator import LocalGenerator

logger = logging.getLogger(__name__)


def generate_answer(query: str, top_k: int) -> str:
    """
    Wrapper around LocalGenerator.
    """
    logger.info(f"Searching for top {top_k} results..")
    results = search_documents(query, top_k=top_k)

    if not results:
        return "No relevant information found."

    generator = LocalGenerator()
    context_chunks = [r["text"] for r in results]
    logger.info("Generating answer..")
    return generator.generate_answer(query, context_chunks)

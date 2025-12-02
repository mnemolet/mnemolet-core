from __future__ import annotations

from dataclasses import dataclass

from mnemolet.core.query.retrieval.search_documents import search_documents
from mnemolet.core.utils.utils import filter_by_min_score


@dataclass
class RetrieverConfig:
    qdrant_url: str
    collection_name: str
    embed_model: str
    top_k: int
    min_score: float


class Retriever:
    def __init__(self, config: RetrieverConfig):
        self.cfg = config

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve and filter context chunks from Qdrant.
        """
        results = search_documents(
            self.cfg.qdrant_url,
            self.cfg.collection_name,
            self.cfg.embed_model,
            query,
            self.cfg.top_k,
        )
        return filter_by_min_score(results, self.cfg.min_score)

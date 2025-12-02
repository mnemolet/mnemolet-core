from mnemolet.core.query.generation.generate_answer import generate_answer


class ChatSession:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embed_model: str,
        ollama_url: str,
        ollama_model: str,
        top_k: int,
        min_score: float,
    ):
        self.history = []
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.top_k = top_k
        self.min_score = min_score
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embed_model = embed_model

    def ask(self, query: str):
        final_results = []

        for chunk, sources in generate_answer(
            qdrant_url=self.qdrant_url,
            collection_name=self.collection_name,
            embed_model=self.embed_model,
            ollama_url=self.ollama_url,
            model=self.ollama_model,
            query=query,
            top_k=self.top_k,
            min_score=self.min_score,
            chat=True,
        ):
            if sources is None:
                # live streaming
                yield chunk
                final_results.append(chunk)

        print()

        # save full response in history
        answer = "".join(final_results)
        self.history.append(
            {"user": query, "assistant": answer, "sources": sources or []}
        )

        return answer

from mnemolet.core.query.generation.generate_answer import generate_answer
from mnemolet.core.query.retrieval.retriever import Retriever


class ChatSession:
    def __init__(
        self,
        retriever: Retriever,
        ollama_url: str,
        ollama_model: str,
    ):
        self.history = []
        self.retriever = retriever
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def ask(self, query: str):
        results = []

        for chunk, sources in generate_answer(
            retriever=self.retriever,
            ollama_url=self.ollama_url,
            model=self.ollama_model,
            query=query,
            chat=True,
        ):
            if sources is None:
                # live streaming
                yield chunk
                results.append(chunk)

        print()

        # save full response in history
        answer = "".join(results)
        self.history.append(
            {"user": query, "assistant": answer, "sources": sources or []}
        )

        return answer

from mnemolet.core.query.generation.generate_answer import generate_answer
from mnemolet.core.query.generation.local_generator import LocalGenerator
from mnemolet.core.query.retrieval.retriever import Retriever


class ChatSession:
    def __init__(
        self,
        retriever: Retriever,
        generator: LocalGenerator,
    ):
        self.history = []
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str):
        results = []

        for chunk, sources in generate_answer(
            retriever=self.retriever,
            generator=self.generator,
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

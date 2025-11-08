from fastapi import FastAPI

from mnemolet_core.query.generation.generate_answer import generate_answer
from mnemolet_core.query.retrieval.search_documents import search_documents

app = FastAPI(title="MnemoLet API", version="0.0.1")


@app.get("/search")
def search(query: str, top_k: int = 3):
    """
    Search documents in Qdrant.
    """
    results = search_documents(query, top_k=top_k)
    return {"results": results}


@app.get("/answer")
def answer(query: str, top_k: int = 3):
    """
    Generate answer from local LLM.
    """
    answer = generate_answer(query, top_k)
    return {"response": answer}
